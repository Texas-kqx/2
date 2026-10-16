# -*- coding: utf-8 -*-
"""step5_clip_zeroshot.py —— 手册第 5 节：CLIP 思想实验。

前面第 4 节的模型是这样的：

    image + text ──► 融合 ──► 分类器 ──► 类别编号

而 CLIP 及之后的多模态模型是这样的：

    image ──► 图像向量 ┐
                        ├──► 余弦相似度最大者 = 匹配
    text  ──► 文本向量 ┘

**没有分类器了**。类别不是被「输出」的，而是通过「和类别描述的相似度」被**比较**出来的。

本脚本验证手册 5.1 表格里的三种能力：
    1. 零样本分类   —— 类别名写成提示词，比相似度，直接在 100 类上算准确率
       （注意：CLIP 从来没在 CUB 上训练过，却不需要任何训练就能分类）
    2. 图文检索     —— image→text / text→image 的 R@1 / R@5
    3. 提示词敏感性 —— 换一种提示词写法，零样本准确率会变多少
       （这是 CLIP 论文里有名的 prompt engineering 现象）

用法：
    python src/step5_clip_zeroshot.py
    python src/step5_clip_zeroshot.py --max-samples 200      # 冒烟
    python src/step5_clip_zeroshot.py --model openai/clip-vit-base-patch32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_ROOT, DATASET_DIR, RESULTS_DIR, FIGURE_DIR,
                    NUM_CLASSES, SEED, get_device, describe_device, set_seed,
                    class_to_prompt, class_to_display)
from dataset import load_annotations, split_json
from train_utils import log_to_file, save_json, compute_metrics, summarize_metrics, markdown_table


# ----------------------------------------------------------------- 提示词模板
# 手册 5.1：类别名写成提示词 -> 比相似度。
# 不同写法会让准确率明显变化，所以这里做一组对照。
PROMPT_TEMPLATES = [
    ("基础", "a photo of a {}"),
    ("加限定", "a photo of a {}, a type of bird"),
    ("加细节", "a clear photo of a {} bird with visible plumage"),
    ("多模板集成", None),      # 特殊处理：下面 ENSEMBLE 里的模板取平均
]

ENSEMBLE_TEMPLATES = [
    "a photo of a {}",
    "a photo of a {}, a type of bird",
    "a clear photo of a {} bird with visible plumage",
    "a blurry photo of a {}",
    "a photo of a {} in its natural habitat",
]


def build_class_texts(classes):
    """返回 {模板名: [100 条提示词]}；「多模板集成」返回多组提示词（外层是模板）。"""
    out = {}
    names = [class_to_display(c) for c in classes]
    for tname, tmpl in PROMPT_TEMPLATES:
        if tmpl is None:
            out[tname] = [[t.format(n) for n in names] for t in ENSEMBLE_TEMPLATES]
        else:
            out[tname] = [tmpl.format(n) for n in names]
    return out


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="手册第 5 节：CLIP 零样本 + 图文检索")
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--tag", default="clip_vit_base_patch32")
    args = ap.parse_args()

    set_seed(SEED)
    device = get_device()

    with log_to_file("step5_clip_zeroshot.log"):
        print("=" * 78)
        print("手册第 5 节  CLIP 思想实验：从「分类器」到「共同语义空间」")
        print("=" * 78)
        print("设备: {}".format(describe_device()))
        print("模型: {}".format(args.model))
        print("HF_ENDPOINT: {}".format(os.environ.get("HF_ENDPOINT")))

        from transformers import CLIPModel, CLIPProcessor
        t0 = time.time()
        model = CLIPModel.from_pretrained(args.model).to(device).eval()
        processor = CLIPProcessor.from_pretrained(args.model)
        print("加载模型用时 {:.1f}s   参数量 {:,}".format(
            time.time() - t0, sum(p.numel() for p in model.parameters())))
        print("投影后的共同空间维度 embed_dim = {}".format(model.config.projection_dim))

        # ---------------- 数据 ----------------
        json_file = split_json(args.split)
        samples, classes = load_annotations(json_file)
        if args.max_samples:
            samples = samples[:args.max_samples]
        if classes is None:
            with open(os.path.join(DATASET_DIR, "annotations.json"), "r", encoding="utf-8") as f:
                classes = json.load(f)["classes"]
        print("\n划分: {}   样本数: {}   类别数: {}".format(args.split, len(samples), len(classes)))

        texts_by_tmpl = build_class_texts(classes)

        # ---------------- 1) 零样本分类 ----------------
        print("\n" + "-" * 78)
        print("① 零样本分类（CLIP 从未在 CUB 上训练过）")
        print("-" * 78)

        # 先把 100 类 × 模板 的文本向量算好
        def encode_texts(text_list, batch=100):
            feats = []
            with torch.no_grad():
                for i in range(0, len(text_list), batch):
                    inputs = processor(text=text_list[i:i + batch], return_tensors="pt",
                                       padding=True, truncation=True).to(device)
                    f = model.get_text_features(**inputs)
                    feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
            return torch.cat(feats)

        text_feat_cache = {}
        for tname, tmpl in PROMPT_TEMPLATES:
            if tmpl is None:
                # 多模板集成：每个模板各算一组文本向量，最后归一化平均
                per = [encode_texts(t) for t in texts_by_tmpl[tname]]
                stacked = torch.stack(per, dim=0).mean(dim=0)
                text_feat_cache[tname] = torch.nn.functional.normalize(stacked, dim=-1)
            else:
                text_feat_cache[tname] = encode_texts(texts_by_tmpl[tname])
        print("文本侧已编码: {} 组提示词，每组 {} 类".format(
            len(text_feat_cache), NUM_CLASSES))
        print("示例提示词: {}".format(texts_by_tmpl["基础"][:3]))

        # 图像向量
        img_feats, labels = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(samples), args.batch):
                chunk = samples[i:i + args.batch]
                imgs = [Image.open(os.path.join(DATA_ROOT, s["image_path"])).convert("RGB")
                        for s in chunk]
                inputs = processor(images=imgs, return_tensors="pt").to(device)
                f = model.get_image_features(**inputs)
                img_feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
                labels.extend(int(s["label"]) for s in chunk)
        img_feats = torch.cat(img_feats)
        labels = np.array(labels)
        print("图像侧已编码: {} 张，用时 {:.1f}s".format(len(labels), time.time() - t0))

        zero_shot = {}
        for tname, _ in PROMPT_TEMPLATES:
            sim = img_feats @ text_feat_cache[tname].T          # [N, 100] 余弦相似度
            preds = sim.argmax(dim=1).numpy()
            m = compute_metrics(labels, preds, NUM_CLASSES)
            m["top5_accuracy"] = float(
                (torch.topk(sim, 5, dim=1).indices.numpy() == labels[:, None]).any(axis=1).mean())
            zero_shot[tname] = m
            print("  提示词 [{:10s}] Top-1={:.4f}  Top-5={:.4f}  宏F1={:.4f}".format(
                tname, m["accuracy"], m["top5_accuracy"], m["f1_macro"]))

        best_tname = max(zero_shot, key=lambda k: zero_shot[k]["accuracy"])
        print("\n零样本最佳提示词: [{}]  Top-1={:.4f}".format(
            best_tname, zero_shot[best_tname]["accuracy"]))
        print("对比参考（第 4 节训练出来的融合模型）见 results/step4_fusion_ablation.json")

        # 说明「没有分类器」这件事：打印某张图的相似度排名
        print("\n「类别不是被输出，而是被比较出来」的现场证据（取第 1 张测试图）：")
        sim0 = (img_feats[0] @ text_feat_cache["基础"].T).numpy()
        order = np.argsort(-sim0)[:5]
        print("  真实类别: {}".format(class_to_display(classes[labels[0]])))
        for r, idx in enumerate(order):
            print("    相似度第{}名 {:.4f}  {}".format(
                r + 1, sim0[idx], class_to_display(classes[idx])))

        # ---------------- 2) 图文检索 ----------------
        print("\n" + "-" * 78)
        print("② 图文检索 R@1 / R@5")
        print("-" * 78)
        # 候选文本池 = 每张图的 10 条描述
        cand_texts, cand_owner = [], []
        for i, s in enumerate(samples):
            for c in s["captions"]:
                cand_texts.append(c)
                cand_owner.append(i)
        cand_owner = np.array(cand_owner)
        print("候选文本池: {} 条（{} 张图 × 10 条描述）".format(len(cand_texts), len(samples)))

        t0 = time.time()
        with torch.no_grad():
            tf = []
            for i in range(0, len(cand_texts), 256):
                inputs = processor(text=cand_texts[i:i + 256], return_tensors="pt",
                                   padding=True, truncation=True).to(device)
                f = model.get_text_features(**inputs)
                tf.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        cand_feats = torch.cat(tf)
        print("文本池编码完成，用时 {:.1f}s".format(time.time() - t0))

        sim_it = (img_feats @ cand_feats.T).numpy()             # [N, 10N] image -> text

        def retrieval_at_k(sim, correct_mask, k, chunk=256):
            """R@k。分块 + argpartition 计算：
            1172 x 11720 的矩阵直接做 argsort(-sim) 会产生几百 MB 的临时数组，
            实测在 Windows 上会把进程直接打崩（0xC0000409），所以必须分块。
            """
            n = sim.shape[0]
            hits = 0
            for s in range(0, n, chunk):
                sub = sim[s:s + chunk]
                kk = min(k, sub.shape[1])
                idx = np.argpartition(-sub, kth=kk - 1, axis=1)[:, :kk]
                rows = np.arange(sub.shape[0])[:, None]
                hits += int(correct_mask[s:s + chunk][rows, idx].any(axis=1).sum())
            return float(hits) / max(1, n)

        # 正样本：候选文本属于该图
        correct = (cand_owner[None, :] == np.arange(len(samples))[:, None])

        r1 = retrieval_at_k(sim_it, correct, 1)
        r5 = retrieval_at_k(sim_it, correct, 5)
        # 随机基线：一张图对应 10 条正样本 / 11720 条候选
        print("  image -> text : R@1={:.4f}  R@5={:.4f}   （随机基线 R@1≈{:.4f}）".format(
            r1, r5, 10.0 / len(cand_texts)))

        # text -> image：转置后端太大，直接按「每条候选文本」分块算
        r1_t = retrieval_at_k(sim_it.T, correct.T, 1)
        r5_t = retrieval_at_k(sim_it.T, correct.T, 5)
        print("  text -> image : R@1={:.4f}  R@5={:.4f}   （随机基线 R@1≈{:.4f}）".format(
            r1_t, r5_t, 1.0 / len(samples)))

        # ---------------- 3) 落盘 ----------------
        summary = {
            "step": "5 CLIP 零样本 + 检索",
            "model": args.model,
            "split": args.split,
            "num_samples": len(samples),
            "embed_dim": int(model.config.projection_dim),
            "zero_shot": {k: {kk: vv for kk, vv in v.items()
                              if kk not in ("confusion_matrix", "per_class_acc",
                                            "per_class_support")}
                          for k, v in zero_shot.items()},
            "zero_shot_best_template": best_tname,
            "retrieval": {"image_to_text_R@1": r1, "image_to_text_R@5": r5,
                          "text_to_image_R@1": r1_t, "text_to_image_R@5": r5_t,
                          "candidate_texts": len(cand_texts)},
            "example_similarity": {
                "true_class": class_to_display(classes[labels[0]]),
                "top5": [{"class": class_to_display(classes[i]), "similarity": float(sim0[i])}
                         for i in order]},
        }
        save_json(summary, os.path.join(RESULTS_DIR, "step5_clip_{}.json".format(args.tag)))

        # 混淆矩阵留给报告用
        np.savez_compressed(
            os.path.join(RESULTS_DIR, "step5_clip_details.npz"),
            confusion_matrix=np.asarray(zero_shot[best_tname]["confusion_matrix"]),
            labels=labels,
            image_feats=img_feats.numpy().astype(np.float16),
            text_feats=text_feat_cache[best_tname].numpy().astype(np.float16))

        print("\n" + "=" * 78)
        print("汇总")
        print("=" * 78)
        rows = [["零样本 Top-1", "{:.4f}".format(zero_shot[best_tname]["accuracy"])],
                ["零样本 Top-5", "{:.4f}".format(zero_shot[best_tname]["top5_accuracy"])],
                ["image→text R@1", "{:.4f}".format(r1)],
                ["image→text R@5", "{:.4f}".format(r5)],
                ["text→image R@1", "{:.4f}".format(r1_t)],
                ["text→image R@5", "{:.4f}".format(r5_t)]]
        print(markdown_table(["指标", "数值"], rows))
        print("\n已保存: results/step5_clip_{}.json".format(args.tag))


if __name__ == "__main__":
    main()
