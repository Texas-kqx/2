# -*- coding: utf-8 -*-
"""dataset.py —— 三个 Dataset：Image-Only / Text-Only / 图文联合。

三者读的是 **同一份标注 JSON**（data_prep.py 生成的 annotations_*.json），
保证「单模态」和「多模态」两阶段看到的数据切分完全一致。

标注 JSON 结构（每个 split 一个文件）：
    {"annotations": [{"image_path": "images/001.../x.jpg",
                      "captions": ["...", ×10],
                      "label": 0}, ...]}
其中 image_path 相对 config.DATA_ROOT。
"""
from __future__ import annotations

import json
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import (DATA_ROOT, IMG_SIZE, MAX_TEXT_LEN, IMAGENET_MEAN,
                    IMAGENET_STD, NUM_CAPTIONS, DATASET_DIR)


# ----------------------------------------------------------------- 工具
def load_annotations(json_file: str):
    """读取标注文件，返回 (samples, classes)。兼容两种写法。"""
    with open(json_file, "r", encoding="utf-8") as f:
        obj = json.load(f)
    samples = obj.get("annotations") or obj.get("splits") or []
    classes = obj.get("classes")
    return samples, classes


def split_json(split: str) -> str:
    """'train' -> <datasets>/annotations_train.json（文件不存在则回落到 annotations.json）"""
    p = os.path.join(DATASET_DIR, "annotations_{}.json".format(split))
    if os.path.exists(p):
        return p
    return os.path.join(DATASET_DIR, "annotations.json")


def build_image_transform(train: bool, img_size: int = IMG_SIZE):
    """图像预处理。训练集做温和增强，验证/测试集只做缩放+中心裁剪。"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_image(rel_path: str) -> Image.Image:
    img = Image.open(os.path.join(DATA_ROOT, rel_path))
    return img.convert("RGB")


# ----------------------------------------------------------------- 基类
class _BaseBirdDataset(Dataset):
    def __init__(self, json_file: str, train: bool = False):
        self.samples, self.classes = load_annotations(json_file)
        self.train = train
        if not self.samples:
            raise RuntimeError("标注文件为空：{}".format(json_file))

    def __len__(self):
        return len(self.samples)

    def _pick_caption(self, item) -> str:
        """一张图有 10 条描述：训练时随机取一条（相当于文本增强），
        评估时固定取第 1 条（保证可复现）。"""
        caps = item["captions"]
        if not caps:
            return ""
        if self.train:
            return caps[torch.randint(0, len(caps), (1,)).item()]
        return caps[0]


# ----------------------------------------------------------------- 2.1 图像
class BirdImageDataset(_BaseBirdDataset):
    """Image-Only：只返回 (image, label)。"""

    def __init__(self, json_file: str, train: bool = False, img_size: int = IMG_SIZE):
        super().__init__(json_file, train)
        self.transform = build_image_transform(train, img_size)

    def __getitem__(self, index):
        item = self.samples[index]
        return (self.transform(_load_image(item["image_path"])),
                int(item["label"]))


# ----------------------------------------------------------------- 2.2 文本
class BirdTextDataset(_BaseBirdDataset):
    """Text-Only：返回 (input_ids, attention_mask, label)。

    对应手册提供的 02_singleModal_TextOnly.py，接口保持一致。
    """

    def __init__(self, json_file: str, tokenizer, train: bool = False,
                 max_length: int = MAX_TEXT_LEN):
        super().__init__(json_file, train)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.samples[index]
        enc = self.tokenizer(self._pick_caption(item),
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


# ----------------------------------------------------------------- 多模态
class BirdMultiModalDataset(_BaseBirdDataset):
    """图文联合：返回 (image, input_ids, attention_mask, label)。

    四种融合方式共用这一份数据 —— 因为融合方式的不同只体现在 fusion.py，
    数据侧完全一致（这也是消融实验能做得干净的前提）。
    """

    def __init__(self, json_file: str, tokenizer, train: bool = False,
                 img_size: int = IMG_SIZE, max_length: int = MAX_TEXT_LEN):
        super().__init__(json_file, train)
        self.transform = build_image_transform(train, img_size)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.samples[index]
        enc = self.tokenizer(self._pick_caption(item),
                             padding="max_length",
                             truncation=True,
                             max_length=self.max_length,
                             return_tensors="pt")
        return {
            "image": self.transform(_load_image(item["image_path"])),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
        }


# ----------------------------------------------------------------- 自检
if __name__ == "__main__":
    img_ds = BirdImageDataset(split_json("train"), train=True)
    print("Image 训练集: {} 张, 类别数 {}".format(len(img_ds), len(img_ds.classes)))
    x, y = img_ds[0]
    print("  单样本: image {} label {}".format(tuple(x.shape), y))

    ev = BirdImageDataset(split_json("test"), train=False)
    x, y = ev[0]
    print("Image 测试集: {} 张, 单样本 image {}".format(len(ev), tuple(x.shape)))
