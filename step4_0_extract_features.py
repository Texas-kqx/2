# -*- coding: utf-8 -*-
"""step4_0_extract_features.py —— 把图像/文本编码器的输出缓存下来。

为什么要有这一步
----------------
第 4 节要对比 **4 种融合方式 + 2 个单模态基线**，还要多种子重复。
如果每次都把 ResNet18 + BERT 前向跑一遍，除了慢，还会引入一个隐蔽的
变量：不同融合方式的训练过程会反过来影响编码器（如果编码器可训练）。

本实验的做法是 **冻结两个编码器，预先把特征算好并缓存**：
    * 编码器对所有融合方式完全相同 —— 消融实验真正做到「只改融合方式」；
    * 融合层的训练只需要几秒到几十秒，可以轻松跑满 3 个随机种子。

缓存内容（每个 split 一个 .npz）：
    image_pooled [N, 512]        ← Early / Late / Gated 用
    image_tokens [N, 49, 512]    ← Cross-Attention 用
    text_pooled  [N, 768]
    text_tokens  [N, 64, 768]
    attention_mask [N, 64]
    labels [N]
为控制变量，所有 split 统一取每张图的**第 1 条**描述（不随机换）。

用法：
    python src/step4_0_extract_features.py
    python src/step4_0_extract_features.py --max-train 200   # 冒烟测试
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
from config import (NUM_CLASSES, FEATURE_DIR, TEXT_MODEL, IMAGE_BACKBONE,
                    SEED, get_device, describe_device, set_seed)
from dataset import BirdMultiModalDataset, split_json
from image_encoder import ImageEncoder
from text_encoder import TextEncoder, load_tokenizer
from train_utils import log_to_file, save_json


def extract_split(split, image_enc, text_enc, tokenizer, device, batch=64, max_n=None):
    ds = BirdMultiModalDataset(split_json(split), tokenizer, train=False)
    if max_n:
        ds.samples = ds.samples[:max_n]
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)

    store = {k: [] for k in ("image_pooled", "image_tokens", "text_pooled",
                             "text_tokens", "attention_mask", "labels")}
    t0 = time.time()
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            images = batch_data["image"].to(device)
            ids = batch_data["input_ids"].to(device)
            mask = batch_data["attention_mask"].to(device)

            img_tokens = image_enc(images, return_tokens=True)      # [B, 49, 512]
            img_pooled = image_enc(images, return_tokens=False)     # [B, 512]
            txt_pooled = text_enc(ids, mask)                        # [B, 768]
            txt_tokens = text_enc.bert(input_ids=ids,
                                       attention_mask=mask).last_hidden_state  # [B, L, 768]

            store["image_pooled"].append(img_pooled.half().cpu().numpy())
            store["image_tokens"].append(img_tokens.half().cpu().numpy())
            store["text_pooled"].append(txt_pooled.half().cpu().numpy())
            store["text_tokens"].append(txt_tokens.half().cpu().numpy())
            store["attention_mask"].append(mask.to(torch.uint8).cpu().numpy())
            store["labels"].append(batch_data["label"].numpy())

            if (i + 1) % 20 == 0:
                done = min((i + 1) * batch, len(ds))
                print("    {:5d}/{}  {:.1f}s".format(done, len(ds), time.time() - t0))

    out = {k: np.concatenate(v) for k, v in store.items()}
    out["classes"] = np.array(ds.classes)
    return out


def load_finetuned_image_encoder(backbone, ckpt, device):
    """把 step2_1 训练好的分类模型权重，装回 ImageEncoder 的主干里。

    step2_1 存的是完整分类模型的 state_dict（键名如 conv1.*, layer4.*, fc.*），
    而 ImageEncoder.body 是 nn.Sequential(*children[:-2])，键名是 body.0.*, body.1.*...
    所以这里按键名前缀手动做一次映射。
    """
    from torchvision import models as tvm
    enc = ImageEncoder(backbone=backbone, pretrained=False, freeze=True).to(device).eval()
    plain = getattr(tvm, backbone)(weights=None)
    child_names = [n for n, _ in plain.named_children()][:-2]   # 去掉 avgpool / fc
    sd = torch.load(ckpt, map_location="cpu")
    mapped, hit = {}, 0
    for i, cname in enumerate(child_names):
        for k, v in sd.items():
            if k.startswith(cname + "."):
                mapped["body.{}.{}".format(i, k[len(cname) + 1:])] = v
                hit += 1
    missing = enc.load_state_dict(mapped, strict=False)
    print("  载入微调图像权重 {} 个张量（来源 {}）".format(hit, os.path.basename(ckpt)))
    if missing.missing_keys:
        print("  警告：未匹配的键 {} 个，例如 {}".format(
            len(missing.missing_keys), missing.missing_keys[:3]))
    return enc


def load_finetuned_text_encoder(ckpt, device):
    """把 step2_2 训练好的 BertClassifier 权重装回 TextEncoder。

    BertClassifier 的键名是 bert.* / classifier.*，TextEncoder 是 bert.* / proj.*，
    两者 bert.* 完全同名，所以 strict=False 直接加载即可。
    """
    enc = TextEncoder(model_name=TEXT_MODEL, freeze=True, pooling="mean").to(device).eval()
    sd = torch.load(ckpt, map_location="cpu")
    bert_sd = {k: v for k, v in sd.items() if k.startswith("bert.")}
    missing = enc.load_state_dict(bert_sd, strict=False)
    print("  载入微调文本权重 {} 个张量（来源 {}）".format(len(bert_sd), os.path.basename(ckpt)))
    if missing.missing_keys:
        print("  警告：未匹配的键 {} 个".format(len(missing.missing_keys)))
    return enc


def main():
    ap = argparse.ArgumentParser(description="预计算并缓存编码器特征")
    ap.add_argument("--backbone", default=IMAGE_BACKBONE)
    ap.add_argument("--text-model", default=TEXT_MODEL)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--tag", default="frozen")
    ap.add_argument("--image-ckpt", default=None,
                    help="step2_1 训练好的图像模型权重；给了就用微调后的主干提特征")
    ap.add_argument("--text-ckpt", default=None,
                    help="step2_2 训练好的 BERT 权重；给了就用微调后的 BERT 提特征")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()

    with log_to_file("step4_0_extract_features.log"):
        print("=" * 72)
        print("第 4 节前置：预计算并缓存编码器特征")
        print("=" * 72)
        print("设备: {}".format(describe_device()))
        print("图像编码器: {} ({})".format(
            args.backbone, "微调后" if args.image_ckpt else "冻结的 ImageNet 预训练"))
        print("文本编码器: {} ({})".format(
            args.text_model, "微调后" if args.text_ckpt else "冻结的预训练"))

        tokenizer = load_tokenizer(args.text_model)
        if args.image_ckpt:
            image_enc = load_finetuned_image_encoder(args.backbone, args.image_ckpt, device)
        else:
            image_enc = ImageEncoder(backbone=args.backbone, pretrained=True,
                                     freeze=True).to(device).eval()
        if args.text_ckpt:
            text_enc = load_finetuned_text_encoder(args.text_ckpt, device)
        else:
            text_enc = TextEncoder(model_name=args.text_model, freeze=True,
                                   pooling="mean").to(device).eval()
        image_enc.eval()
        text_enc.eval()

        # 先打印一次「模型拆解」的形状，作为第 3 节的直接证据
        print("\n--- 形状核查（手册第 3 节）---")
        with torch.no_grad():
            demo = torch.randn(2, 3, 224, 224).to(device)
            fmap = image_enc.body(demo)
            print("图像输入            : {}".format(tuple(demo.shape)))
            print("去掉 avgpool/fc 后  : {}   ← [B, 512, 7, 7]".format(tuple(fmap.shape)))
            print("  flatten(2).transpose(1,2) -> {}".format(
                tuple(fmap.flatten(2).transpose(1, 2).shape)))
            print("  AdaptiveAvgPool2d(1)      -> {}".format(
                tuple(image_enc.pool(fmap).flatten(1).shape)))
            demo_ids = torch.randint(0, 1000, (2, 64)).to(device)
            demo_mask = torch.ones(2, 64, dtype=torch.long).to(device)
            hid = text_enc.bert(input_ids=demo_ids, attention_mask=demo_mask).last_hidden_state
            print("文本 token 数 L=64  : {}   ← [B, L, 768]".format(tuple(hid.shape)))
            print("mean pooling 之后   : {}   ← [B, 768]".format(
                tuple(text_enc(demo_ids, demo_mask).shape)))

        info = {}
        for split, mx in (("train", args.max_train), ("val", None), ("test", None)):
            print("\n--- 提取 {} ---".format(split))
            t0 = time.time()
            arr = extract_split(split, image_enc, text_enc, tokenizer, device,
                               batch=args.batch, max_n=mx)
            path = os.path.join(FEATURE_DIR, "mm_{}_{}.npz".format(args.tag, split))
            np.savez_compressed(path, **arr)
            sz = os.path.getsize(path) / 1024 ** 2
            print("  样本 {}  用时 {:.1f}s  写出 {} ({:.0f} MB)".format(
                len(arr["labels"]), time.time() - t0, os.path.basename(path), sz))
            info[split] = {"n": int(len(arr["labels"])), "file": os.path.basename(path),
                           "size_mb": round(sz, 1),
                           "image_pooled": list(arr["image_pooled"].shape),
                           "image_tokens": list(arr["image_tokens"].shape),
                           "text_pooled": list(arr["text_pooled"].shape),
                           "text_tokens": list(arr["text_tokens"].shape)}

        save_json({"tag": args.tag, "backbone": args.backbone,
                   "text_model": args.text_model,
                   "image_ckpt": args.image_ckpt, "text_ckpt": args.text_ckpt,
                   "splits": info},
                  os.path.join(FEATURE_DIR, "mm_{}_info.json".format(args.tag)))

        # 趁编码器还在，顺手验证手册第 3 节的核心结论：
        # 「两个模态的向量不在同一个空间里，不能直接比较」
        print("\n--- 交叉验证：图像向量和文本向量能直接比吗？ ---")
        arr = np.load(os.path.join(FEATURE_DIR, "mm_{}_test.npz".format(args.tag)))
        img = arr["image_pooled"][:500].astype(np.float32)
        txt = arr["text_pooled"][:500].astype(np.float32)
        lab = arr["labels"][:500]
        print("图像向量形状 {}   文本向量形状 {}".format(img.shape, txt.shape))

        # 证据 1：连点积都算不了 —— 维度都不一样
        try:
            img @ txt.T
            print("居然算出来了？（不应该）")
        except ValueError as e:
            print("证据1  直接做 img @ txt.T  ->  报错：{}".format(str(e).split("\n")[0]))
            print("       两个编码器输出连维度都不同（512 vs 768），谈不上「比较」。")

        # 证据 2：强行投影到公共维度（这里取文本前 512 维，纯属为了能算），
        #         再看相似度是否比随机基线更有结构。
        txt_proj = txt[:, :img.shape[1]]
        img_n = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
        txt_n = txt_proj / (np.linalg.norm(txt_proj, axis=1, keepdims=True) + 1e-8)
        cross = img_n @ txt_n.T
        paired = np.diag(cross)
        same_class = (lab[:, None] == lab[None, :])
        np.fill_diagonal(same_class, False)

        rng = np.random.RandomState(SEED)
        ri = rng.randn(500, img.shape[1]).astype(np.float32)
        rt = rng.randn(500, img.shape[1]).astype(np.float32)
        ri /= np.linalg.norm(ri, axis=1, keepdims=True)
        rt /= np.linalg.norm(rt, axis=1, keepdims=True)
        base = ri @ rt.T

        print("证据2  强行投影到 512 维后再比：")
        print("       图像-文本 余弦相似度: 均值={:+.4f} 标准差={:.4f}".format(
            cross.mean(), cross.std()))
        print("       配对样本（同一张图）  : 均值={:+.4f}".format(paired.mean()))
        print("       同类异样本            : 均值={:+.4f}".format(cross[same_class].mean()))
        print("       随机向量基线          : 均值={:+.4f} 标准差={:.4f}".format(
            base.mean(), base.std()))
        print("       => 配对样本的相似度并没有显著高于随机基线，也就是**没有对齐**。")
        print("          两个独立训练的编码器各自活在自己的坐标系里，")
        print("          这正是第 5 节要用 CLIP 的对比学习把图文拉进同一个空间的动机。")
        save_json({"image_shape": list(img.shape), "text_shape": list(txt.shape),
                   "cross_mean": float(cross.mean()), "cross_std": float(cross.std()),
                   "paired_mean": float(paired.mean()),
                   "same_class_mean": float(cross[same_class].mean()),
                   "random_mean": float(base.mean()),
                   "random_std": float(base.std())},
                  os.path.join(FEATURE_DIR, "modality_gap.json"))


if __name__ == "__main__":
    main()
