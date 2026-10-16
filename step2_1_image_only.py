# -*- coding: utf-8 -*-
"""step2_1_image_only.py —— 手册 2.1 节：Image-Only 训练。

输入：图像 + 类别标签
模型：ResNet18 / MobileNetV2（ImageNet 预训练 + 替换分类头）
输出：类别

手册本节的学习目标：
    1. 实现训练、测试和分类性能指标验证
    2. 「把模型拆开」，查看对图像处理后的特征

所以这个脚本做两件事：
    (A) 完整训练 + 验证 + 测试，输出 accuracy / macro-P / macro-R / macro-F1、
        混淆矩阵、训练曲线；
    (B) 训练结束后，用 image_encoder.ImageEncoder 把主干「拆开」，
        导出倒数第二层的特征图 [B,512,7,7]，验证它如何变成
        全局向量 [B,512] 和 49 个视觉 token [B,49,512]，
        并把特征存成 .npy 供第 3 节做 t-SNE 可视化。

用法：
    python src/step2_1_image_only.py                       # 默认 resnet18
    python src/step2_1_image_only.py --backbone mobilenet_v2
    python src/step2_1_image_only.py --epochs 5 --tag smoke   # 快速冒烟
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
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (NUM_CLASSES, RESULTS_DIR, FEATURE_DIR, FIGURE_DIR,
                    IMAGE_EPOCHS, IMAGE_BATCH, IMAGE_LR, SEED,
                    get_device, describe_device, set_seed, class_to_display)
from dataset import BirdImageDataset, split_json, build_image_transform
from train_utils import (log_to_file, History, compute_metrics, summarize_metrics,
                         save_json, plot_confusion_matrix, confusion_pairs,
                         markdown_table)


# ----------------------------------------------------------------- 模型
WEIGHT_ENUM = {
    "resnet18": ("resnet18", "ResNet18_Weights", "IMAGENET1K_V1"),
    "resnet34": ("resnet34", "ResNet34_Weights", "IMAGENET1K_V1"),
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V1"),
    "mobilenet_v2": ("mobilenet_v2", "MobileNet_V2_Weights", "IMAGENET1K_V1"),
}


def build_model(backbone: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False):
    """构造分类模型：ImageNet 预训练主干 + 换成 num_classes 的新分类头。"""
    if backbone not in WEIGHT_ENUM:
        raise ValueError("支持 {}，收到 {}".format(list(WEIGHT_ENUM), backbone))
    ctor_name, weights_name, enum_name = WEIGHT_ENUM[backbone]
    ctor = getattr(models, ctor_name)

    weights = None
    if pretrained:
        weights_cls = getattr(models, weights_name, None)
        if weights_cls is not None:
            weights = getattr(weights_cls, enum_name)
    net = ctor(weights=weights)

    # 换掉分类头
    if backbone.startswith("resnet"):
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, num_classes)
    else:  # mobilenet_v2: classifier = [Dropout, Linear]
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in net.named_parameters():
            is_head = name.startswith("fc.") or name.startswith("classifier.")
            p.requires_grad = is_head
    return net


def count_params(net):
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------- 训练/评估
def run_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": correct / max(1, total),
        "logits": np.concatenate(all_logits) if all_logits else np.zeros((0, 0)),
        "labels": np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=int),
    }


def main():
    ap = argparse.ArgumentParser(description="手册 2.1 Image-Only 训练")
    ap.add_argument("--backbone", default="resnet18", choices=list(WEIGHT_ENUM))
    ap.add_argument("--epochs", type=int, default=IMAGE_EPOCHS)
    ap.add_argument("--batch", type=int, default=IMAGE_BATCH)
    ap.add_argument("--lr", type=float, default=IMAGE_LR)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="只训练分类头（线性探针），用于验证预训练特征的质量")
    ap.add_argument("--no-pretrained", action="store_true", help="从零训练（随机初始化）")
    ap.add_argument("--tag", default=None, help="结果文件后缀，默认用 backbone 名")
    ap.add_argument("--no-features", action="store_true", help="跳过特征导出")
    args = ap.parse_args()

    tag = args.tag or (args.backbone + ("_frozen" if args.freeze_backbone else "")
                       + ("_scratch" if args.no_pretrained else ""))
    set_seed(SEED)
    device = get_device()

    with log_to_file("step2_1_image_only_{}.log".format(tag)):
        print("=" * 72)
        print("手册 2.1  Image-Only 训练    backbone={}  tag={}".format(args.backbone, tag))
        print("=" * 72)
        print("设备: {}".format(describe_device()))
        print("预训练: {}   冻结主干: {}   训练轮数: {}   batch: {}   lr: {}".format(
            not args.no_pretrained, args.freeze_backbone, args.epochs, args.batch, args.lr))

        # ---------------- 数据 ----------------
        train_ds = BirdImageDataset(split_json("train"), train=True)
        val_ds = BirdImageDataset(split_json("val"), train=False)
        test_ds = BirdImageDataset(split_json("test"), train=False)
        print("样本数  train={}  val={}  test={}  类别数={}".format(
            len(train_ds), len(val_ds), len(test_ds), len(train_ds.classes)))

        nw = 0 if os.name == "nt" else 4
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=nw, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                                num_workers=nw, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                                 num_workers=nw, pin_memory=True)

        # ---------------- 模型 ----------------
        model = build_model(args.backbone, NUM_CLASSES,
                            pretrained=not args.no_pretrained,
                            freeze_backbone=args.freeze_backbone).to(device)
        total_p, train_p = count_params(model)
        print("模型参数量: 总 {:,}  可训练 {:,}".format(total_p, train_p))

        criterion = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # ---------------- 训练 ----------------
        hist = History("step2_1_image_only_{}".format(tag))
        best_acc, best_state, best_epoch = -1.0, None, -1
        print("\n开始训练...")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            va = run_epoch(model, val_loader, criterion, None, device, train=False)
            scheduler.step()
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
                torch.save(best_state, os.path.join(RESULTS_DIR, "best_image_{}.pth".format(tag)))

        print("\n最佳验证集准确率: {:.4f}  (第 {} 轮)".format(best_acc, best_epoch))
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---------------- 测试 ----------------
        te = run_epoch(model, test_loader, criterion, None, device, train=False)
        metrics = compute_metrics(te["labels"], te["logits"].argmax(axis=1), NUM_CLASSES)
        print("\n" + "-" * 72)
        print("测试集结果: {}".format(summarize_metrics(metrics)))
        print("-" * 72)

        classes = train_ds.classes
        worst = np.argsort(metrics["per_class_acc"])[:10]
        print("\n最差的 10 个类别（准确率）：")
        rows = []
        for i in worst:
            rows.append([classes[i], "{:.3f}".format(metrics["per_class_acc"][i]),
                         metrics["per_class_support"][i]])
        print(markdown_table(["类别", "准确率", "样本数"], rows))
        print("\n最容易混淆的类别对：")
        rows = [[c["true"], c["pred"], c["count"]] for c in confusion_pairs(metrics, classes, 10)]
        print(markdown_table(["真实类别", "被预测为", "次数"], rows))

        # ---------------- 落盘 ----------------
        hist.save()
        hist.plot(title="Image-Only {} 训练曲线".format(args.backbone))
        plot_confusion_matrix(metrics, classes, "step2_1_{}".format(tag))
        summary = {
            "step": "2.1 Image-Only",
            "backbone": args.backbone,
            "pretrained": not args.no_pretrained,
            "freeze_backbone": args.freeze_backbone,
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "params_total": total_p,
            "params_trainable": train_p,
            "best_val_accuracy": best_acc,
            "best_epoch": best_epoch,
            "avg_epoch_sec": float(np.mean([r["epoch_sec"] for r in hist.records])),
            "test": metrics,
            "confusing_pairs": confusion_pairs(metrics, classes, 15),
        }
        save_json(summary, os.path.join(RESULTS_DIR, "step2_1_image_only_{}.json".format(tag)))
        torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "final_image_{}.pth".format(tag)))
        print("\n已保存: results/step2_1_image_only_{}.json".format(tag))

        # ---------------- (B) 拆开模型看特征 ----------------
        if not args.no_features:
            print("\n" + "=" * 72)
            print("把模型拆开：查看图像经过主干之后变成了什么")
            print("=" * 72)
            analyze_features(model, args.backbone, test_loader, device, tag, classes)

        print("\n总耗时 {:.1f} 分钟".format(hist.elapsed / 60))


# ----------------------------------------------------------------- 特征拆解
def analyze_features(model, backbone, test_loader, device, tag, classes):
    """验证 [B,3,224,224] -> [B,512,7,7] -> [B,512] / [B,49,512] 这条链路。"""
    model.eval()
    # 取主干（去掉 avgpool 与 fc）：resnet 的 children 最后两层是 avgpool/fc
    if backbone.startswith("resnet"):
        body = nn.Sequential(*list(model.children())[:-2])
    else:
        # mobilenet_v2: features -> avgpool -> classifier
        body = model.features

    images, labels = next(iter(test_loader))
    images = images.to(device)
    with torch.no_grad():
        fmap = body(images)

    print("输入图像张量      : {}".format(tuple(images.shape)))
    print("主干输出特征图    : {}   ← 224/32 = 7".format(tuple(fmap.shape)))
    pooled = fmap.mean(dim=(2, 3))
    tokens = fmap.flatten(2).transpose(1, 2)
    print("全局平均池化后    : {}   ← Early / Late / Gated 融合用".format(tuple(pooled.shape)))
    print("展平成 token 序列 : {}   ← Attention 融合用（每张图 49 个「视觉单词」）".format(
        tuple(tokens.shape)))
    print("特征图统计        : mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
        fmap.mean().item(), fmap.std().item(), fmap.min().item(), fmap.max().item()))

    # 导出一小批特征供第 3 节 t-SNE 用
    n_dump = min(1500, len(test_loader.dataset))
    feats, labs, imgs_dump = [], [], []
    with torch.no_grad():
        got = 0
        for images, labels in test_loader:
            images = images.to(device)
            f = body(images).mean(dim=(2, 3))
            take = min(n_dump - got, images.size(0))
            feats.append(f[:take].float().cpu().numpy())
            labs.append(labels[:take].numpy())
            imgs_dump.append(images[:take].cpu())
            got += take
            if got >= n_dump:
                break
    feats = np.concatenate(feats)
    labs = np.concatenate(labs)
    np.savez_compressed(os.path.join(FEATURE_DIR, "image_feats_{}.npz".format(tag)),
                        features=feats, labels=labs, classes=np.array(classes))
    print("已导出 {} 个图像特征到 results/features/image_feats_{}.npz".format(len(feats), tag))

    # 同类别 vs 不同类别的余弦相似度：说明「特征空间里同类确实更近」
    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim = fn @ fn.T
    same = labs[:, None] == labs[None, :]
    np.fill_diagonal(same, False)
    print("同类样本平均余弦相似度  : {:.4f}".format(sim[same].mean()))
    print("异类样本平均余弦相似度  : {:.4f}".format(sim[~same].mean()))
    return pooled.shape, tokens.shape


if __name__ == "__main__":
    main()
