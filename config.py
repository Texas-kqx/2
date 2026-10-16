# -*- coding: utf-8 -*-
"""config.py —— 全实验统一配置。

把路径、随机种子、划分比例、超参数集中在这里，
保证「单模态 / 编码器拆解 / 融合消融 / CLIP」四个阶段用的是
**同一份数据划分**、**同一个随机种子**（控制变量）。
"""
from __future__ import annotations

import os
import random

# ---------------------------------------------------------------- 路径
# 原始数据（类别文件夹 + 同名 txt）
DATA_ROOT = os.environ.get("BIRD_DATA_ROOT", r"D:\cxdownload\CUB_200_subset")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")     # 生成的 annotations*.json
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")      # 指标 / 日志 / 权重
FEATURE_DIR = os.path.join(RESULTS_DIR, "features")      # 预计算特征缓存
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")        # 图表
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

for _d in (DATASET_DIR, RESULTS_DIR, FEATURE_DIR, FIGURE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# 预训练权重 / HF 模型缓存放在仓库内，不写系统盘
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".torch_cache"))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
# 国内网络走镜像（若已能直连 huggingface.co 可注释掉）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------------------------------------------------------------- 数据
NUM_CLASSES = 100
NUM_CAPTIONS = 10          # 每张图 10 条描述
SPLIT_RATIO = (0.6, 0.2, 0.2)   # 手册要求 6:2:2
SEED = 42
IMG_SIZE = 224
MAX_TEXT_LEN = 64

# ImageNet 标准化（ResNet 预训练要求）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- 模型
IMAGE_BACKBONE = "resnet18"      # resnet18 -> [B, 512, 7, 7]
TEXT_MODEL = "bert-base-uncased"  # -> [B, L, 768]
IMAGE_FEAT_DIM = 512
TEXT_FEAT_DIM = 768

# ---------------------------------------------------------------- 训练超参
IMAGE_EPOCHS = 15
IMAGE_BATCH = 32
IMAGE_LR = 1e-4

TEXT_EPOCHS = 10
TEXT_BATCH = 16
TEXT_LR = 2e-5

FUSION_EPOCHS = 40
FUSION_BATCH = 128
FUSION_LR = 1e-3
FUSION_D_MODEL = 256
FUSION_HEADS = 4

# ---------------------------------------------------------------- 工具
def set_seed(seed: int = SEED) -> None:
    """同时固定 python / numpy / torch 的随机源。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0 ({}) 显存 {:.1f} GB".format(
            torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
    return "cpu"


# 类别文件夹名 -> 提示词，供 CLIP 零样本使用
def class_to_prompt(class_name: str) -> str:
    """'001.Black_footed_Albatross' -> 'a photo of a black footed albatross'"""
    name = class_name.split(".", 1)[-1].replace("_", " ").strip().lower()
    return "a photo of a {}".format(name)


def class_to_display(class_name: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'（报告里用）"""
    return class_name.split(".", 1)[-1].replace("_", " ")
