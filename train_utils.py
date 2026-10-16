# -*- coding: utf-8 -*-
"""train_utils.py —— 训练日志、指标计算、结果落盘与绘图。

所有 step 脚本共用这套工具，保证：
  * 指标口径一致（accuracy / macro-P / macro-R / macro-F1）；
  * 每个实验都能把 stdout 同时写进 results/logs/*.log；
  * 训练曲线、混淆矩阵统一存到 results/figures/。
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from config import LOG_DIR, FIGURE_DIR, RESULTS_DIR


# ----------------------------------------------------------------- 日志
class Tee(object):
    """把 stdout 同时写到控制台和日志文件。"""

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class log_to_file(object):
    """with log_to_file("results/logs/step1.log"): ..."""

    def __init__(self, name):
        self.path = os.path.join(LOG_DIR, name)

    def __enter__(self):
        self.tee = Tee(self.path)
        sys.stdout = self.tee
        return self.path

    def __exit__(self, *exc):
        sys.stdout = self.tee.stdout
        self.tee.close()
        return False


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------- 指标
def compute_metrics(y_true, y_pred, num_classes=None):
    """返回统一口径的分类指标。

    使用 sklearn 计算（缺库时回退到纯 numpy 实现，保证脚本不会因为
    环境差异跑不起来）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(num_classes))

    try:
        from sklearn.metrics import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        r = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    except ImportError:                                   # pragma: no cover
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, q in zip(y_true, y_pred):
            cm[int(t), int(q)] += 1
        acc = float(np.trace(cm)) / max(1, cm.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.diag(cm) / np.maximum(cm.sum(axis=0), 1)
            rec = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
            f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        p, r, f1 = float(np.nanmean(prec)), float(np.nanmean(rec)), float(np.nanmean(f1s))

    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    support = cm.sum(axis=1)
    return {
        "accuracy": float(acc),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "per_class_acc": per_class_acc.tolist(),
        "per_class_support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "num_classes": int(num_classes),
        "num_samples": int(len(y_true)),
    }


def topk_accuracy(logits, labels, k=5):
    """top-k 准确率（logits: [N, C]）。"""
    logits = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1, 1)
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == labels).any(axis=1).mean())


def summarize_metrics(m):
    """把指标字典压成一行可打印文本。"""
    return ("acc={:.4f}  P={:.4f}  R={:.4f}  F1={:.4f}  (n={})".format(
        m["accuracy"], m["precision_macro"], m["recall_macro"],
        m["f1_macro"], m["num_samples"]))


def confusion_pairs(metrics, classes, top=10):
    """返回混淆最严重的 (真实类, 预测成, 次数) 列表。"""
    cm = np.asarray(metrics["confusion_matrix"])
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), i, j))
    pairs.sort(reverse=True)
    out = []
    for n, i, j in pairs[:top]:
        out.append({"count": n,
                    "true": classes[i] if classes else str(i),
                    "pred": classes[j] if classes else str(j)})
    return out


# ----------------------------------------------------------------- 训练历史
class History(object):
    """逐 epoch 记录指标，结束后一键落盘 + 画曲线。"""

    def __init__(self, name):
        self.name = name
        self.records = []
        self.start = time.time()

    def add(self, epoch, **kw):
        rec = {"epoch": epoch}
        rec.update(kw)
        self.records.append(rec)

    def best(self, key="val_accuracy", mode="max"):
        if not self.records:
            return None
        f = max if mode == "max" else min
        return f(self.records, key=lambda r: r.get(key, -1 if mode == "max" else 1e9))

    @property
    def elapsed(self):
        return time.time() - self.start

    def save(self, path=None):
        path = path or os.path.join(RESULTS_DIR, "{}_history.json".format(self.name))
        save_json({"name": self.name, "elapsed_sec": self.elapsed,
                   "records": self.records}, path)
        return path

    def plot(self, keys=("loss", "val_loss"), title=None, path=None):
        """画训练曲线（matplotlib 缺失时静默跳过）。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        if not self.records:
            return None
        path = path or os.path.join(FIGURE_DIR, "{}_curve.png".format(self.name))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        epochs = [r["epoch"] for r in self.records]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for k in keys:
            if all(k in r for r in self.records):
                axes[0].plot(epochs, [r[k] for r in self.records], marker="o", ms=3, label=k)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        for k in ("accuracy", "val_accuracy"):
            if all(k in r for r in self.records):
                axes[1].plot(epochs, [r[k] for r in self.records], marker="o", ms=3, label=k)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.suptitle(title or self.name)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def plot_confusion_matrix(metrics, classes, name, normalize=True, max_classes=100):
    """画混淆矩阵热力图。类别太多时默认归一化，便于看对角线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    path = os.path.join(FIGURE_DIR, "{}_confusion.png".format(name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("{} (100 classes{})".format(name, ", row-normalized" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def markdown_table(headers, rows):
    """生成 Markdown 表格文本。"""
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
