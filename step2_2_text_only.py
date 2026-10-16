# -*- coding: utf-8 -*-
"""step2_2_text_only.py —— 手册 2.2 节：Text-Only 训练。

输入：一条描述文本 + 类别标签
模型：BERT-base（对应手册附带的 02_singleModal_TextOnly.py 的实现路线）
输出：类别

说明：手册 2.2 的标题里写的是「模型：CLIP」，但给出的参考代码
02_singleModal_TextOnly.py 用的是 **BERT + 分类头**。本脚本采用后者的路线
（也是「单模态分类」的正确做法）；CLIP 的用法放到第 5 节
「从分类器到共同语义空间」里做零样本与检索，两节各司其职、不重复。

在原 02 脚本基础上补齐了这些（原脚本只有训练 + 测试，没有验证集）：
  * 每个 epoch 在验证集上评估，保存验证集最佳权重（模型选择）；
  * 测试集输出 accuracy / macro-P / macro-R / macro-F1 与混淆矩阵；
  * 训练历史、日志、指标全部落盘到 results/；
  * --pooling {mean,cls}：手册第 3 节讲的两种句向量池化方式，可对比；
  * --max-captions N：一张图有 10 条描述，训练时可每轮随机换一条（文本增强）。

用法：
    python src/step2_2_text_only.py
    python src/step2_2_text_only.py --pooling cls --epochs 5
    python src/step2_2_text_only.py --freeze --tag bert_frozen
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (NUM_CLASSES, RESULTS_DIR, TEXT_MODEL, TEXT_EPOCHS, TEXT_BATCH,
                    TEXT_LR, MAX_TEXT_LEN, SEED, get_device, describe_device, set_seed)
from dataset import BirdTextDataset, split_json
from train_utils import (log_to_file, History, compute_metrics, summarize_metrics,
                         save_json, plot_confusion_matrix, confusion_pairs,
                         markdown_table, topk_accuracy)


# ----------------------------------------------------------------- 模型
class BertClassifier(nn.Module):
    """BERT + 线性分类头（与手册 02 脚本同构）。

    关键点：BERT 输出 [B, L, 768]，分类需要一个「句子向量」，
    这里实现了手册第 3 节讲的两种池化：
        mean —— attention_mask 加权平均（默认，更稳）
        cls  —— 取 [CLS] 位置的向量
    """

    def __init__(self, model_name=TEXT_MODEL, num_classes=NUM_CLASSES,
                 pooling="mean", dropout=0.1, freeze=False, pretrained=True):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.hidden_size = self.bert.config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.freeze = freeze
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        """冻结时把 BERT 固定在 eval 模式（否则 dropout/BN 仍在变）。"""
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def pool(self, hidden, attention_mask):
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)     # [B, L, 1]
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, return_embedding=False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sent = self.pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(sent))
        if return_embedding:
            return logits, sent
        return logits


# ----------------------------------------------------------------- 训练/评估
def run_epoch(model, loader, criterion, optimizer, device, train=True, scheduler=None):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_labels, all_emb = [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits, sent = model(ids, mask, return_embedding=True)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    # OneCycleLR 是**逐 step** 的调度器：不在每个 batch 后调用，
                    # 学习率会永远停在初始值 max_lr/div_factor（div_factor 默认 25），
                    # 也就是比预期小 25 倍 —— 这是本实验踩过的最隐蔽的一个坑。
                    scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_emb.append(sent.detach().float().cpu().numpy())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "logits": np.concatenate(all_logits),
        "labels": np.concatenate(all_labels),
        "embeddings": np.concatenate(all_emb),
    }


def main():
    ap = argparse.ArgumentParser(description="手册 2.2 Text-Only 训练（BERT）")
    ap.add_argument("--model", default=TEXT_MODEL)
    ap.add_argument("--epochs", type=int, default=TEXT_EPOCHS)
    ap.add_argument("--batch", type=int, default=TEXT_BATCH)
    ap.add_argument("--lr", type=float, default=TEXT_LR)
    ap.add_argument("--head-lr", type=float, default=1e-3,
                    help="分类头的学习率（判别式微调：头 > BERT 主干）")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=MAX_TEXT_LEN)
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--freeze", action="store_true", help="冻结 BERT，只训分类头")
    ap.add_argument("--no-pretrained", action="store_true", help="BERT 随机初始化（对照）")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or ("bert_{}{}".format(args.pooling, "_frozen" if args.freeze else "")
                       + ("_scratch" if args.no_pretrained else ""))
    set_seed(SEED)
    device = get_device()

    with log_to_file("step2_2_text_only_{}.log".format(tag)):
        print("=" * 72)
        print("手册 2.2  Text-Only 训练    model={}  tag={}".format(args.model, tag))
        print("=" * 72)
        print("设备: {}".format(describe_device()))
        print("HF_ENDPOINT = {}".format(os.environ.get("HF_ENDPOINT")))
        print("池化方式: {}   冻结 BERT: {}   轮数: {}   batch: {}   lr: {}".format(
            args.pooling, args.freeze, args.epochs, args.batch, args.lr))

        # ---------------- 分词器 ----------------
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        demo = "a small blue bird with a long beak"
        enc = tokenizer(demo, return_tensors="pt")
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        print("\n分词演示：\"{}\"".format(demo))
        print("  token 序列: {}".format(toks))
        print("  token id   : {}".format(enc["input_ids"][0].tolist()))
        print("  词表大小   : {}".format(tokenizer.vocab_size))

        # ---------------- 数据 ----------------
        train_ds = BirdTextDataset(split_json("train"), tokenizer, train=True,
                                   max_length=args.max_length)
        val_ds = BirdTextDataset(split_json("val"), tokenizer, train=False,
                                 max_length=args.max_length)
        test_ds = BirdTextDataset(split_json("test"), tokenizer, train=False,
                                  max_length=args.max_length)
        print("\n样本数  train={}  val={}  test={}".format(
            len(train_ds), len(val_ds), len(test_ds)))
        print("注意：训练时每轮从该图的 10 条描述里随机取 1 条（文本增强），"
              "验证/测试固定取第 1 条。")

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

        # ---------------- 模型 ----------------
        model = BertClassifier(model_name=args.model, num_classes=NUM_CLASSES,
                               pooling=args.pooling, freeze=args.freeze,
                               pretrained=not args.no_pretrained).to(device)
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("模型参数量: 总 {:,}  可训练 {:,}".format(total_p, train_p))

        criterion = nn.CrossEntropyLoss()
        # 判别式微调（discriminative fine-tuning）：BERT 主干用小学习率，
        # 随机初始化的分类头用大学习率。两者共用一个 2e-5 会有一个后果：
        # 分类头收敛太慢，整个模型看起来「学不动」。
        head_params, bert_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (head_params if name.startswith("classifier.") else bert_params).append(p)
        groups = []
        if bert_params:
            groups.append({"params": bert_params, "lr": args.lr})
        if head_params:
            groups.append({"params": head_params, "lr": args.head_lr})
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
        total_steps = max(1, len(train_loader) * args.epochs)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in groups],
            total_steps=total_steps, pct_start=0.1)
        print("学习率: BERT {:.1e}   分类头 {:.1e}   (OneCycle, 逐 step 调度)".format(
            args.lr, args.head_lr))

        # ---------------- 训练 ----------------
        hist = History("step2_2_text_only_{}".format(tag))
        best_acc, best_state, best_epoch = -1.0, None, -1
        print("\n开始训练...")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, criterion, optimizer, device,
                           train=True, scheduler=scheduler)
            va = run_epoch(model, val_loader, criterion, None, device, train=False)
            dt = time.time() - t0
            hist.add(epoch, loss=tr["loss"], accuracy=tr["accuracy"],
                     val_loss=va["loss"], val_accuracy=va["accuracy"],
                     lr=optimizer.param_groups[0]["lr"], epoch_sec=dt)
            print("Epoch [{:2d}/{:2d}]  train_loss={:.4f} train_acc={:.4f} | "
                  "val_loss={:.4f} val_acc={:.4f} | {:.1f}s".format(
                      epoch, args.epochs, tr["loss"], tr["accuracy"],
                      va["loss"], va["accuracy"], dt))
            if va["accuracy"] > best_acc:
                best_acc, best_epoch = va["accuracy"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state,
                           os.path.join(RESULTS_DIR, "best_text_{}.pth".format(tag)))

        print("\n最佳验证集准确率: {:.4f}  (第 {} 轮)".format(best_acc, best_epoch))
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---------------- 测试 ----------------
        te = run_epoch(model, test_loader, criterion, None, device, train=False)
        preds = te["logits"].argmax(axis=1)
        metrics = compute_metrics(te["labels"], preds, NUM_CLASSES)
        metrics["top5_accuracy"] = topk_accuracy(te["logits"], te["labels"], 5)
        print("\n" + "-" * 72)
        print("测试集结果: {}".format(summarize_metrics(metrics)))
        print("Top-5 准确率: {:.4f}".format(metrics["top5_accuracy"]))
        print("-" * 72)

        classes = train_ds.classes
        worst = np.argsort(metrics["per_class_acc"])[:10]
        print("\n最差的 10 个类别（说明「只靠文字」有多难）：")
        rows = [[classes[i], "{:.3f}".format(metrics["per_class_acc"][i]),
                 metrics["per_class_support"][i]] for i in worst]
        print(markdown_table(["类别", "准确率", "样本数"], rows))
        print("\n最容易混淆的类别对：")
        rows = [[c["true"], c["pred"], c["count"]] for c in confusion_pairs(metrics, classes, 10)]
        print(markdown_table(["真实类别", "被预测为", "次数"], rows))

        # ---------------- 落盘 ----------------
        hist.save()
        hist.plot(title="Text-Only BERT ({}) 训练曲线".format(args.pooling))
        plot_confusion_matrix(metrics, classes, "step2_2_{}".format(tag))
        summary = {
            "step": "2.2 Text-Only (BERT)",
            "model": args.model,
            "pretrained": not args.no_pretrained,
            "freeze_bert": args.freeze,
            "pooling": args.pooling,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "max_length": args.max_length,
            "params_total": total_p,
            "params_trainable": train_p,
            "best_val_accuracy": best_acc,
            "best_epoch": best_epoch,
            "avg_epoch_sec": float(np.mean([r["epoch_sec"] for r in hist.records])),
            "test": metrics,
            "confusing_pairs": confusion_pairs(metrics, classes, 15),
        }
        save_json(summary, os.path.join(RESULTS_DIR, "step2_2_text_only_{}.json".format(tag)))
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "final_text_{}.pth".format(tag)))

        # 句向量导出（第 3 节做 t-SNE 与「两个模态空间不通用」的验证）
        np.savez_compressed(
            os.path.join(RESULTS_DIR, "features", "text_feats_{}.npz".format(tag)),
            features=te["embeddings"], labels=te["labels"], classes=np.array(classes))
        print("已保存句向量: results/features/text_feats_{}.npz".format(tag))
        print("\n总耗时 {:.1f} 分钟".format(hist.elapsed / 60))


if __name__ == "__main__":
    main()
