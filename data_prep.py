# -*- coding: utf-8 -*-
"""data_prep.py —— 数据准备（手册 1.3 节）。

做四件事：
  1. 扫描 `images/` 与 `text/` 两个平行目录，按 **文件名主干** 配成图文对；
  2. 体检：统计孤儿文件（有图无文 / 有文无图）、每类样本数、caption 行数；
  3. 按 **6 : 2 : 2 分层划分**（保证每个类别的比例一致），随机种子固定；
  4. 生成两种标注格式：
       * datasets/annotations.json         —— 手册 1.3 节要求的单文件格式
                                              (meta / classes / class_prompts / splits)
       * datasets/annotations_{split}.json —— 02_singleModal_TextOnly.py 直接可读
                                              的 {"annotations": [...]} 格式

用法：
    python src/data_prep.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_ROOT, DATASET_DIR, RESULTS_DIR, NUM_CAPTIONS,
                    SPLIT_RATIO, SEED, class_to_prompt)

IMG_EXTS = (".jpg", ".jpeg", ".png")


def scan_pairs():
    """返回 (pairs, report)。

    pairs: [{"image_path": "images/001.../x.jpg",
             "text_path":  "text/001.../x.txt",
             "captions":   [...], "label": 0, "class_name": "001..."}]
    image_path / text_path 都是相对 DATA_ROOT 的 POSIX 风格路径。
    """
    img_root = os.path.join(DATA_ROOT, "images")
    txt_root = os.path.join(DATA_ROOT, "text")
    if not os.path.isdir(img_root):
        raise SystemExit("找不到图片目录：{}".format(img_root))

    classes = sorted(d for d in os.listdir(img_root)
                     if os.path.isdir(os.path.join(img_root, d)))
    pairs = []
    orphan_img, orphan_txt, bad_caption = [], [], []

    for label, cname in enumerate(classes):
        cdir = os.path.join(img_root, cname)
        stems = OrderedDict()
        for fn in sorted(os.listdir(cdir)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMG_EXTS:
                stems[stem] = fn

        tdir = os.path.join(txt_root, cname)
        txt_stems = set()
        if os.path.isdir(tdir):
            for fn in sorted(os.listdir(tdir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() == ".txt":
                    txt_stems.add(stem)

        for stem, img_fn in stems.items():
            if stem not in txt_stems:
                orphan_img.append("{}/{}".format(cname, img_fn))
                continue
            txt_path = os.path.join(tdir, stem + ".txt")
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                caps = [ln.strip() for ln in f if ln.strip()]
            if len(caps) != NUM_CAPTIONS:
                bad_caption.append({"file": "{}/{}.txt".format(cname, stem),
                                    "lines": len(caps)})
            pairs.append({
                "image_path": "images/{}/{}".format(cname, img_fn),
                "text_path": "text/{}/{}.txt".format(cname, stem),
                "captions": caps,
                "label": label,
                "class_name": cname,
            })

        for stem in txt_stems:
            if stem not in stems:
                orphan_txt.append("{}/{}.txt".format(cname, stem))

    report = {
        "data_root": DATA_ROOT,
        "num_classes": len(classes),
        "num_pairs": len(pairs),
        "num_orphan_image": len(orphan_img),
        "num_orphan_text": len(orphan_txt),
        "num_bad_caption": len(bad_caption),
        "orphan_images": orphan_img,
        "orphan_texts": orphan_txt,
        "bad_captions": bad_caption,
        "per_class": {},
    }
    for cname in classes:
        n = sum(1 for p in pairs if p["class_name"] == cname)
        report["per_class"][cname] = n
    return classes, pairs, report


def stratified_split(pairs, classes, ratio=SPLIT_RATIO, seed=SEED):
    """按类别分层做 6:2:2 划分。

    对每个类别内部独立打乱再切分，因此每个类别在 train/val/test 中的
    比例都约为 6:2:2（小类别至少保证 val / test 各 1 张）。
    """
    rng = random.Random(seed)
    by_class = {c: [] for c in classes}
    for p in pairs:
        by_class[p["class_name"]].append(p)

    splits = {"train": [], "val": [], "test": []}
    for cname in classes:
        items = by_class[cname]
        idx = list(range(len(items)))
        rng.shuffle(idx)
        n = len(items)
        n_train = int(round(n * ratio[0]))
        n_val = int(round(n * ratio[1]))
        # 保证小类别三类都非空
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        n_test = n - n_train - n_val
        for k, i in enumerate(idx):
            if k < n_train:
                splits["train"].append(items[i])
            elif k < n_train + n_val:
                splits["val"].append(items[i])
            else:
                splits["test"].append(items[i])
        assert n_test >= 0
    for k in splits:
        splits[k].sort(key=lambda d: (d["label"], d["image_path"]))
    return splits


def main():
    print("=" * 70)
    print("数据准备：{}".format(DATA_ROOT))
    print("=" * 70)

    classes, pairs, report = scan_pairs()
    print("类别数        : {}".format(report["num_classes"]))
    print("可用图文对    : {}".format(report["num_pairs"]))
    print("有图无文(丢弃): {}".format(report["num_orphan_image"]))
    print("有文无图      : {}".format(report["num_orphan_text"]))
    print("caption 行数异常: {}".format(report["num_bad_caption"]))
    if report["orphan_images"]:
        print("  ↓ 被丢弃的孤儿图片：")
        for p in report["orphan_images"]:
            print("     - {}".format(p))
    if report["bad_captions"]:
        for b in report["bad_captions"][:10]:
            print("     ! {} 只有 {} 行".format(b["file"], b["lines"]))

    counts = list(report["per_class"].values())
    print("每类样本数    : min={} max={} avg={:.1f}".format(
        min(counts), max(counts), sum(counts) / len(counts)))

    splits = stratified_split(pairs, classes)
    for k in ("train", "val", "test"):
        print("  {:5s}: {:5d} 条".format(k, len(splits[k])))

    class_prompts = {c: class_to_prompt(c) for c in classes}

    # ---------- 格式 1：手册 1.3 节的单文件 annotations.json ----------
    def slim(item):
        return {"image_path": item["image_path"],
                "captions": item["captions"],
                "label": item["label"],
                "class_name": item["class_name"]}

    annotations = {
        "meta": {
            "source": "CUB_200_2011 subset（100 类，图像 + 10 条描述）",
            "data_root": DATA_ROOT,
            "num_classes": len(classes),
            "num_captions": NUM_CAPTIONS,
            "num_samples": len(pairs),
            "split_ratio": list(SPLIT_RATIO),
            "seed": SEED,
        },
        "classes": classes,
        "class_prompts": class_prompts,
        "splits": {k: [slim(x) for x in v] for k, v in splits.items()},
    }
    p_all = os.path.join(DATASET_DIR, "annotations.json")
    with open(p_all, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=1)
    print("\n已写出 {}".format(p_all))

    # ---------- 格式 2：02_singleModal_TextOnly.py 需要的三个文件 ----------
    for k in ("train", "val", "test"):
        obj = {
            "meta": annotations["meta"],
            "classes": classes,
            "class_prompts": class_prompts,
            "annotations": [slim(x) for x in splits[k]],
        }
        path = os.path.join(DATASET_DIR, "annotations_{}.json".format(k))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        print("已写出 {}".format(path))

    # ---------- 体检报告 ----------
    with open(os.path.join(RESULTS_DIR, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump({"report": report,
                   "split_sizes": {k: len(v) for k, v in splits.items()}},
                  f, ensure_ascii=False, indent=1)
    print("已写出 {}".format(os.path.join(RESULTS_DIR, "data_report.json")))


if __name__ == "__main__":
    main()
