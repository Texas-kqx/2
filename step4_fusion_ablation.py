# -*- coding: utf-8 -*-
"""step4_fusion_ablation.py —— 手册第 4 节：多模态融合与消融实验。

对比对象（手册 4.1 节的四种融合 + 两个单模态基线）：
    image_only   仅图像
    text_only    仅文本
    early        ① 早融合：concat -> MLP
    late         ② 晚融合：两个分类头 + 概率加权（w 可学）
    cross_attn   ③ 跨模态注意力：图像 token 作 Query 去文本 token 里找描述
    gated        ④ 门控融合：逐样本决定更信谁

实验设计（控制变量）
--------------------
* 六个模型用的是 **完全相同的冻结特征**（step4_0 缓存），所以差异只来自融合方式；
* 每个配置跑 **3 个随机种子**，报告 均值±标准差 —— 避免把抽样噪声当成结论；
* 训练/验证/测试用同一套 6:2:2 划分；
* 统一在验证集上做模型选择（early stopping by best val acc），测试集只评一次。

额外导出（给报告用）
--------------------
* late ：学到的 w（文本模态权重）——「模型更信谁」的直接读数；
* gated：测试集上每个样本的门控值分布；
* cross_attn：若干样本的注意力矩阵，可回答「模型在看哪个词」。

用法：
    python src/step4_fusion_ablation.py
    python src/step4_fusion_ablation.py --fusions early,late --seeds 0 --epochs 10   # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (NUM_CLASSES, FEATURE_DIR, RESULTS_DIR, FIGURE_DIR,
                    FUSION_EPOCHS, FUSION_BATCH, FUSION_LR, IMAGE_FEAT_DIM,
                    TEXT_FEAT_DIM, SEED, get_device, describe_device, set_seed)
from fusion import (build_fusion, FUSION_CHOICES, CrossAttentionFusion,
                    LateFusion, GatedFusion)
from train_utils import (log_to_file, compute_metrics, summarize_metrics,
                         save_json, markdown_table, topk_accuracy,
                         plot_confusion_matrix)


# ----------------------------------------------------------------- 数据
class CachedFeatures(object):
    """把 step4_0 缓存的 .npz 装进显存。

    为了省显存，token 序列以 fp16 存放，取 batch 时再转回 fp32
    （混合精度存储、单精度计算，不影响数值结论）。
    """

    def __init__(self, tag, split, device):
        path = os.path.join(FEATURE_DIR, "mm_{}_{}.npz".format(tag, split))
        if not os.path.exists(path):
            raise SystemExit(
                "找不到特征缓存 {}。请先运行：\n"
                "    python src/step4_0_extract_features.py".format(path))
        arr = np.load(path)
        self.device = device
        self.image_pooled = torch.as_tensor(arr["image_pooled"], dtype=torch.float16).to(device)
        self.image_tokens = torch.as_tensor(arr["image_tokens"], dtype=torch.float16).to(device)
        self.text_pooled = torch.as_tensor(arr["text_pooled"], dtype=torch.float16).to(device)
        self.text_tokens = torch.as_tensor(arr["text_tokens"], dtype=torch.float16).to(device)
        self.attention_mask = torch.as_tensor(arr["attention_mask"], dtype=torch.long).to(device)
        self.labels = torch.as_tensor(arr["labels"], dtype=torch.long).to(device)
        self.classes = [str(c) for c in arr["classes"]]

    def __len__(self):
        return int(self.labels.shape[0])

    def batch(self, idx):
        """idx: LongTensor。返回 (image_feat, text_feat, attention_mask, labels)。"""
        image_feat = {"pooled": self.image_pooled[idx].float(),
                      "tokens": self.image_tokens[idx].float()}
        text_feat = {"pooled": self.text_pooled[idx].float(),
                     "tokens": self.text_tokens[idx].float()}
        return image_feat, text_feat, self.attention_mask[idx], self.labels[idx]

    def nbytes(self):
        tot = 0
        for k in ("image_pooled", "image_tokens", "text_pooled", "text_tokens",
                  "attention_mask", "labels"):
            tot += getattr(self, k).numel() * getattr(self, k).element_size()
        return tot


# ----------------------------------------------------------------- 评估
@torch.no_grad()
def evaluate(model, data, batch_size=256, collect=None):
    model.eval()
    n = len(data)
    logits_all = []
    extras = {"gate": [], "weight": [], "attn_weights": [], "attn_index": []}
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n), device=data.device)
        image_feat, text_feat, mask, labels = data.batch(idx)
        out = model(image_feat, text_feat, mask)
        logits_all.append(out["logits"].float().cpu())
        if "gate" in out:
            extras["gate"].append(out["gate"].float().mean(dim=1).cpu())
        if "attn_weights" in out:
            extras["attn_weights"].append(out["attn_weights"].float().cpu())
            extras["attn_index"].append(idx.cpu())
    logits = torch.cat(logits_all).numpy()
    labels = data.labels.cpu().numpy()
    preds = logits.argmax(axis=1)
    metrics = compute_metrics(labels, preds, NUM_CLASSES)
    metrics["top5_accuracy"] = topk_accuracy(logits, labels, 5)
    metrics["logits"] = logits
    metrics["labels"] = labels
    if extras["gate"]:
        g = torch.cat(extras["gate"]).numpy()
        metrics["gate_mean"] = float(g.mean())
        metrics["gate_std"] = float(g.std())
        metrics["gate_values"] = g.tolist()
    if extras["attn_weights"]:
        metrics["attn_weights"] = torch.cat(extras["attn_weights"]).numpy()
        metrics["attn_index"] = torch.cat(extras["attn_index"]).numpy()
    return metrics


# ----------------------------------------------------------------- 训练一个配置
def train_one(name, train_data, val_data, test_data, args, seed, device, verbose=True):
    set_seed(seed)
    model = build_fusion(name, image_dim=IMAGE_FEAT_DIM, text_dim=TEXT_FEAT_DIM,
                         num_classes=NUM_CLASSES,
                         d_model=args.d_model, dropout=args.dropout,
                         attn_dropout=args.attn_dropout).to(device)
    # 训练时关掉注意力权重物化（省时间），评估/可视化时再打开
    if isinstance(model, CrossAttentionFusion):
        model.need_weights = False

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # 先用几个 epoch 线性 warmup，再余弦退火。
    # warmup 对跨模态注意力尤其重要：投影层 + 注意力都是随机初始化的，
    # 一上来就用 1e-3 容易把注意力打散到无法恢复（实测有种子 loss 卡在 ln(100)）。
    warmup = max(1, int(args.epochs * args.warmup_frac))
    sched_warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                   total_iters=warmup)
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[sched_warm, sched_cos], milestones=[warmup])
    criterion = nn.CrossEntropyLoss()

    n = len(train_data)
    steps_per_epoch = max(1, n // args.batch)
    best_val, best_state, best_epoch = -1.0, None, -1
    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            image_feat, text_feat, mask, labels = train_data.batch(idx)
            out = model(image_feat, text_feat, mask)
            loss = criterion(out["logits"], labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_data, collect=False)
            history.append({"epoch": epoch, "loss": total_loss / steps_per_epoch,
                            "val_accuracy": val["accuracy"]})
            if verbose:
                print("      epoch {:3d}/{:3d}  loss={:.4f}  val_acc={:.4f}".format(
                    epoch, args.epochs, total_loss / steps_per_epoch, val["accuracy"]))
            if val["accuracy"] > best_val:
                best_val, best_epoch = val["accuracy"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_sec = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试集只评一次；cross_attn 打开注意力以便导出
    if isinstance(model, CrossAttentionFusion):
        model.need_weights = True
    test = evaluate(model, test_data, collect=True)

    result = {
        "fusion": name,
        "seed": seed,
        "params": n_params,
        "epochs": args.epochs,
        "best_val_accuracy": best_val,
        "best_epoch": best_epoch,
        "train_sec": train_sec,
        "test_accuracy": test["accuracy"],
        "test_precision_macro": test["precision_macro"],
        "test_recall_macro": test["recall_macro"],
        "test_f1_macro": test["f1_macro"],
        "test_top5_accuracy": test["top5_accuracy"],
        "history": history,
    }
    if isinstance(model, LateFusion):
        result["late_weight_text"] = float(model.weight.item())
    if "gate_mean" in test:
        result["gate_mean"] = test["gate_mean"]
        result["gate_std"] = test["gate_std"]
    # 返回测试指标全文，供上层画混淆矩阵/注意力
    result["_test_full"] = test
    result["_model"] = model
    return result


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="手册第 4 节：多模态融合消融实验")
    ap.add_argument("--tag", default="frozen", help="特征缓存前缀（对应 step4_0）")
    ap.add_argument("--fusions", default=",".join(FUSION_CHOICES))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=FUSION_EPOCHS)
    ap.add_argument("--batch", type=int, default=FUSION_BATCH)
    ap.add_argument("--lr", type=float, default=FUSION_LR)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--attn-dropout", type=float, default=0.1,
                    help="跨模态注意力的 dropout，单独控制（不跟随 --dropout）")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值，0=关闭")
    ap.add_argument("--warmup-frac", type=float, default=0.1, help="warmup 占总轮数的比例")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--label", default=None, help="结果里标注用的名字，如 frozen / finetuned")
    args = ap.parse_args()

    fusions = [f.strip() for f in args.fusions.split(",") if f.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = get_device()

    with log_to_file("step4_fusion_ablation.log"):
        print("=" * 78)
        print("手册第 4 节  多模态融合消融实验")
        print("=" * 78)
        print("设备: {}".format(describe_device()))
        print("融合方式: {}".format(fusions))
        print("随机种子: {}".format(seeds))
        print("训练轮数: {}   batch: {}   lr: {}   dropout: {}".format(
            args.epochs, args.batch, args.lr, args.dropout))

        train_data = CachedFeatures(args.tag, "train", device)
        val_data = CachedFeatures(args.tag, "val", device)
        test_data = CachedFeatures(args.tag, "test", device)
        print("\n样本数 train={} val={} test={}".format(
            len(train_data), len(val_data), len(test_data)))
        print("特征缓存占显存: {:.2f} GB".format(
            (train_data.nbytes() + val_data.nbytes() + test_data.nbytes()) / 1024 ** 3))
        print("图像特征维度: pooled={} tokens={}".format(
            tuple(train_data.image_pooled.shape[1:]),
            tuple(train_data.image_tokens.shape[1:])))
        print("文本特征维度: pooled={} tokens={}".format(
            tuple(train_data.text_pooled.shape[1:]),
            tuple(train_data.text_tokens.shape[1:])))

        all_runs = []
        full_metrics = {}
        for name in fusions:
            print("\n" + "=" * 78)
            print("配置: {}".format(name))
            print("=" * 78)
            for seed in seeds:
                print("  --- seed={} ---".format(seed))
                res = train_one(name, train_data, val_data, test_data, args, seed, device)
                full = res.pop("_test_full")
                model = res.pop("_model")
                res["run"] = "{}_{}".format(name, seed)
                all_runs.append(res)
                if seed == seeds[0]:
                    full_metrics[name] = {
                        "logits": full["logits"], "labels": full["labels"],
                        "preds": full["logits"].argmax(axis=1),
                        "confusion_matrix": full["confusion_matrix"],
                    }
                    if "attn_weights" in full:
                        full_metrics[name]["attn_weights"] = full["attn_weights"]
                        full_metrics[name]["attn_index"] = full["attn_index"]
                    if "gate_values" in full:
                        full_metrics[name]["gate_values"] = np.array(full["gate_values"])
                print("      测试: {}".format(summarize_metrics({
                    "accuracy": res["test_accuracy"],
                    "precision_macro": res["test_precision_macro"],
                    "recall_macro": res["test_recall_macro"],
                    "f1_macro": res["test_f1_macro"],
                    "num_samples": len(test_data)})))

        # ---------------- 汇总 ----------------
        print("\n" + "=" * 78)
        print("消融汇总（{} 个种子，均值 ± 标准差）".format(len(seeds)))
        print("=" * 78)
        summary_rows = []
        for name in fusions:
            runs = [r for r in all_runs if r["fusion"] == name]
            accs = np.array([r["test_accuracy"] for r in runs])
            f1s = np.array([r["test_f1_macro"] for r in runs])
            vals = np.array([r["best_val_accuracy"] for r in runs])
            summary_rows.append({
                "fusion": name,
                "params": runs[0]["params"],
                "val_accuracy_mean": float(vals.mean()), "val_accuracy_std": float(vals.std()),
                "test_accuracy_mean": float(accs.mean()), "test_accuracy_std": float(accs.std()),
                "test_f1_mean": float(f1s.mean()), "test_f1_std": float(f1s.std()),
                "top5_mean": float(np.mean([r["test_top5_accuracy"] for r in runs])),
                "train_sec": float(np.mean([r["train_sec"] for r in runs])),
                "n_seeds": len(runs),
            })

        from fusion import FUSION_LABELS_CN
        rows = []
        for s in summary_rows:
            rows.append([
                FUSION_LABELS_CN.get(s["fusion"], s["fusion"]),
                "{:,}".format(s["params"]),
                "{:.4f} ± {:.4f}".format(s["test_accuracy_mean"], s["test_accuracy_std"]),
                "{:.4f}".format(s["test_f1_mean"]),
                "{:.4f}".format(s["top5_mean"]),
                "{:.1f}s".format(s["train_sec"]),
            ])
        table = markdown_table(
            ["融合方式", "参数量", "测试准确率 (mean±std)", "宏 F1", "Top-5", "平均耗时"], rows)
        print(table)

        # 单模态 vs 多模态的提升
        base_img = next((s for s in summary_rows if s["fusion"] == "image_only"), None)
        base_txt = next((s for s in summary_rows if s["fusion"] == "text_only"), None)
        best = max(summary_rows, key=lambda s: s["test_accuracy_mean"])
        print("\n最佳融合方式: {}  测试准确率 {:.4f}".format(
            best["fusion"], best["test_accuracy_mean"]))
        if base_img:
            print("  相对「仅图像」提升: {:+.2f} 个百分点".format(
                (best["test_accuracy_mean"] - base_img["test_accuracy_mean"]) * 100))
        if base_txt:
            print("  相对「仅文本」提升: {:+.2f} 个百分点".format(
                (best["test_accuracy_mean"] - base_txt["test_accuracy_mean"]) * 100))

        # late 的 w
        late_runs = [r for r in all_runs if r["fusion"] == "late"]
        if late_runs:
            ws = np.array([r["late_weight_text"] for r in late_runs])
            print("\n晚融合学到的文本权重 w = {:.4f} ± {:.4f}（图像权重 1-w = {:.4f}）".format(
                ws.mean(), ws.std(), 1 - ws.mean()))
        gated_runs = [r for r in all_runs if r["fusion"] == "gated"]
        if gated_runs:
            gs = np.array([r["gate_mean"] for r in gated_runs])
            print("门控融合的平均门控值 a = {:.4f} ± {:.4f}".format(gs.mean(), gs.std()))

        # ---------------- 落盘 ----------------
        label = args.label or args.tag
        save_json({"tag": args.tag, "label": label, "args": vars(args),
                   "runs": all_runs, "summary": summary_rows,
                   "table_markdown": table},
                  os.path.join(RESULTS_DIR,
                               "step4_fusion_ablation_{}.json".format(args.tag)))

        np.savez_compressed(
            os.path.join(RESULTS_DIR,
                         "step4_fusion_details_{}.npz".format(args.tag)),
            **{k: v for k, v in _flatten_details(full_metrics).items()})

        # 每个融合方式画混淆矩阵（用 seed 0 的那次）
        classes = test_data.classes
        for name in fusions:
            if name in full_metrics:
                plot_confusion_matrix(
                    {"confusion_matrix": full_metrics[name]["confusion_matrix"]},
                    classes, "step4_{}_{}".format(args.tag, name))

        print("\n已保存: results/step4_fusion_ablation_{}.json".format(args.tag))
        print("已保存: results/step4_fusion_details_{}.npz".format(args.tag))
        print("\n完成。")


def _flatten_details(full_metrics):
    """把每个配置的细节数组拍平成 键名 = "<配置>_<字段>"，存进一个 npz。

    这里额外保存了 **逐样本预测 preds** 和 labels，是为了后面能做
    McNemar 配对检验 —— 两个模型在同一份测试集上比较时，
    「准确率差几个千分点」和「这个差异是否显著」是两个问题。
    """
    out = {}
    for name, d in full_metrics.items():
        for k, v in d.items():
            if k in ("confusion_matrix", "attn_weights", "attn_index",
                     "gate_values", "preds", "labels"):
                out["{}_{}".format(name, k)] = np.asarray(v)
    return out


if __name__ == "__main__":
    main()
