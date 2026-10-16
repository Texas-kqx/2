# -*- coding: utf-8 -*-
"""fusion.py —— 多模态融合模块（本实验的核心）。

四种融合方式**接口完全一致**：

    forward(image_feat, text_feat, attention_mask) -> {"logits": [B, C], ...}

其中 image_feat / text_feat 都是一个 dict，同时携带两种表示：

    image_feat = {"pooled": [B, 512],      # 全局平均池化向量，Early/Late/Gated 用
                  "tokens": [B, 49, 512]}  # 49 个空间 token，Cross-Attention 用
    text_feat  = {"pooled": [B, 768],
                  "tokens": [B, L, 768]}

这样设计的原因：手册要求「换融合方式只改配置里的 fusion: 一个字段」。
把两种粒度的特征一次性喂进来，调用侧代码就完全不用动，
消融实验才做得干净。

四种融合的本质区别（手册 4.1 节）：
    ① Early  (ConcatFusion)        —— 拼起来送 MLP，最简单，但没有显式交互
    ② Late   (LateFusion)          —— 各模态先各自分类，再按权重融合概率，可解释
    ③ Cross  (CrossAttentionFusion)—— 图像 token 当 Query 去文本 token 里找描述，可导出注意力
    ④ Gated  (GatedFusion)         —— 逐样本决定「这次更信谁」，能抑制不可靠文本
另外附带两个单模态基线（ImageOnly / TextOnly），用于消融表的对照。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================= 工具
def _mlp(in_dim, out_dim, hidden=None, dropout=0.2, num_layers=1):
    """构造一个简单的 MLP 分类头。hidden=None 时退化为单层线性。"""
    if hidden is None:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden), nn.ReLU(inplace=True)]
    for _ in range(max(0, num_layers - 1)):
        layers += [nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
    layers += [nn.Dropout(dropout), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class BaseFusion(nn.Module):
    """所有融合模块的基类：统一保存维度信息，统一返回 dict。"""

    name = "base"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100, **kw):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.num_classes = num_classes

    # 子类实现
    def fuse(self, image_feat, text_feat, attention_mask=None):
        raise NotImplementedError

    def forward(self, image_feat, text_feat, attention_mask=None):
        out = self.fuse(image_feat, text_feat, attention_mask)
        if "logits" not in out:
            raise RuntimeError("{} 必须返回 logits".format(type(self).__name__))
        return out

    def param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ================================================================= ① Early
class ConcatFusion(BaseFusion):
    """① Early Fusion（早融合）。

        image[512] ┐
                   ├─ concat ─► [1280] ─► MLP ─► 类别
        text [768] ┘

    特点：实现最简单；缺点是两模态只是「被拼在一起」，没有显式交互。
    """

    name = "early"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 hidden=512, dropout=0.3, depth=2, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.classifier = _mlp(image_dim + text_dim, num_classes,
                               hidden=hidden, dropout=dropout, num_layers=depth)

    def fuse(self, image_feat, text_feat, attention_mask=None):
        z = torch.cat([image_feat["pooled"], text_feat["pooled"]], dim=-1)   # [B, 1280]
        return {"logits": self.classifier(z)}


# ================================================================= ② Late
class LateFusion(BaseFusion):
    """② Late Fusion（晚融合）。

        image ─► 图像分类头 ─► p_img ┐
                                      ├─ w·p_text + (1-w)·p_img ─► 类别
        text  ─► 文本分类头 ─► p_text ┘

    特点：任一模态缺失都能工作、可解释（能看到 w）；缺点是模态之间从不交流。
    w 由 sigmoid 参数化，训练中自己学出来 —— 这就是「模型更信谁」的直接读数。

    注意：为了和统一的 CrossEntropyLoss 训练循环兼容，这里的 "logits"
    返回的是 log 概率（CE(log p, y) 在数学上等价于 NLL(p, y)，
    且 log_softmax(log p) == log p，梯度同样正确）。
    """

    name = "late"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 hidden=256, dropout=0.2, init_w=0.5, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.image_head = _mlp(image_dim, num_classes, hidden=hidden, dropout=dropout)
        self.text_head = _mlp(text_dim, num_classes, hidden=hidden, dropout=dropout)
        # w = sigmoid(raw)，raw 初始化到 logit(init_w)，默认从 0.5 开始（不偏袒任何模态）
        import math
        init_w = min(max(init_w, 1e-4), 1 - 1e-4)
        self.raw_w = nn.Parameter(torch.tensor([math.log(init_w / (1 - init_w))],
                                               dtype=torch.float32))

    @property
    def weight(self):
        """文本模态的权重 w（图像模态权重为 1-w）。"""
        return torch.sigmoid(self.raw_w)

    def fuse(self, image_feat, text_feat, attention_mask=None):
        image_logits = self.image_head(image_feat["pooled"])
        text_logits = self.text_head(text_feat["pooled"])
        p_img = F.softmax(image_logits, dim=-1)
        p_text = F.softmax(text_logits, dim=-1)
        w = self.weight
        probs = w * p_text + (1.0 - w) * p_img
        return {"logits": torch.log(probs.clamp_min(1e-8)),
                "probs": probs,
                "image_logits": image_logits,
                "text_logits": text_logits,
                "weight": w.detach()}


# ================================================================= ③ Cross-Attention
class CrossAttentionFusion(BaseFusion):
    """③ Cross-Attention Fusion（跨模态注意力）—— 手册推荐。

        image tokens [49,512] ── Query ┐
                                       ├─ Multi-Head Attention ─► [49,d] ─► 池化 ─► 分类
        text  tokens [L,768]  ── Key/Value ┘

    让「图像的每个空间位置」去文本里找相关描述（"我这里是 blue 吗？"），
    还能导出注意力权重做可视化 —— 可以看到「模型在看哪个词」。

    实现要点：
      * 两个模态维度不同（512 / 768），先用 Linear 投影到公共维度 d；
      * attention_mask 转成 key_padding_mask，让 [PAD] 不参与注意力
        （否则 padding 会稀释真实词的信息，和手册里 mean pooling 的坑一模一样）；
      * need_weights=True 走的是非 fused 的慢路径，但为了拿到可解释的注意力矩阵，
        教学实验里值得。
    """

    name = "cross_attn"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 d_model=256, num_heads=4, dropout=0.1, attn_dropout=0.1,
                 hidden=256, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.d_model = d_model
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        # 注意：attention 的 dropout 单独控制。用分类头那个较大的 dropout（如 0.3）
        # 会让注意力在训练早期就被打散，实测会导致个别种子完全不收敛（loss 卡在 ln(100)）。
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = _mlp(d_model, num_classes, hidden=hidden, dropout=dropout)
        # need_weights=True 会强制走非 fused 的慢路径（因为要物化注意力矩阵）。
        # 训练时把它关掉（只关心梯度），只在推理/可视化时打开。
        self.need_weights = True

    def fuse(self, image_feat, text_feat, attention_mask=None):
        q = self.image_proj(image_feat["tokens"])      # [B, 49, d]
        kv = self.text_proj(text_feat["tokens"])       # [B, L, d]

        key_padding_mask = None
        if attention_mask is not None:
            # True 表示这个位置要被忽略
            key_padding_mask = (attention_mask == 0)
            # 极端情况：整句都被 mask 掉会让 softmax 全 -inf 产生 NaN
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        attn_out, attn_w = self.attn(q, kv, kv,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=self.need_weights,
                                     average_attn_weights=False)
        pooled = self.norm(attn_out).mean(dim=1)       # 对 49 个位置求平均
        out = {"logits": self.classifier(pooled)}
        if self.need_weights:
            out["attn_weights"] = attn_w               # [B, heads, 49, L]
        return out


# ================================================================= ④ Gated
class GatedFusion(BaseFusion):
    """④ Gated Fusion（门控融合）。

        a = σ(W[img; txt])      fusion = a·img + (1-a)·txt

    特点：逐样本决定「这次更信谁」，对「描述没提颜色」的不可靠文本有天然抑制。
    和 Late Fusion 的区别：Late 是一个**全局**标量 w，Gated 是**每个样本**一个 a，
    而且门控是在特征空间做的（信息还没被压成概率）。
    """

    name = "gated"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 d_model=256, dropout=0.1, hidden=256, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.d_model = d_model
        self.image_proj = nn.Linear(image_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )
        self.classifier = _mlp(d_model, num_classes, hidden=hidden, dropout=dropout)

    def fuse(self, image_feat, text_feat, attention_mask=None):
        img = self.image_proj(image_feat["pooled"])    # [B, d]
        txt = self.text_proj(text_feat["pooled"])      # [B, d]
        a = self.gate(torch.cat([img, txt], dim=-1))   # [B, d]
        fused = a * img + (1.0 - a) * txt
        return {"logits": self.classifier(fused),
                "gate": a}                             # 逐样本、逐维度的信任度


# ================================================================= 单模态基线
class ImageOnlyClassifier(BaseFusion):
    """消融对照：只用图像（文本分支完全不用）。"""

    name = "image_only"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 hidden=256, dropout=0.2, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.classifier = _mlp(image_dim, num_classes, hidden=hidden, dropout=dropout)

    def fuse(self, image_feat, text_feat, attention_mask=None):
        return {"logits": self.classifier(image_feat["pooled"])}


class TextOnlyClassifier(BaseFusion):
    """消融对照：只用文本。"""

    name = "text_only"

    def __init__(self, image_dim=512, text_dim=768, num_classes=100,
                 hidden=256, dropout=0.2, **kw):
        super().__init__(image_dim, text_dim, num_classes)
        self.classifier = _mlp(text_dim, num_classes, hidden=hidden, dropout=dropout)

    def fuse(self, image_feat, text_feat, attention_mask=None):
        return {"logits": self.classifier(text_feat["pooled"])}


# ================================================================= 注册表
FUSION_REGISTRY = {
    "early": ConcatFusion,
    "late": LateFusion,
    "cross_attn": CrossAttentionFusion,
    "gated": GatedFusion,
    "image_only": ImageOnlyClassifier,
    "text_only": TextOnlyClassifier,
}

# 手册 4.1 节的四种融合 + 两个单模态基线
FUSION_CHOICES = ["image_only", "text_only", "early", "late", "cross_attn", "gated"]

FUSION_LABELS_CN = {
    "image_only": "仅图像",
    "text_only": "仅文本",
    "early": "① 早融合 (Concat)",
    "late": "② 晚融合 (Late)",
    "cross_attn": "③ 跨模态注意力 (Cross-Attn)",
    "gated": "④ 门控融合 (Gated)",
}


def build_fusion(name, image_dim=512, text_dim=768, num_classes=100, **kw):
    """按名字构造融合模块 —— 这就是「换融合方式只改一个字段」的落点。"""
    if name not in FUSION_REGISTRY:
        raise ValueError("未知融合方式 {}，可选：{}".format(name, list(FUSION_REGISTRY)))
    return FUSION_REGISTRY[name](image_dim=image_dim, text_dim=text_dim,
                                 num_classes=num_classes, **kw)


# ================================================================= 自检
if __name__ == "__main__":
    B, C = 4, 100
    image_feat = {"pooled": torch.randn(B, 512), "tokens": torch.randn(B, 49, 512)}
    text_feat = {"pooled": torch.randn(B, 768), "tokens": torch.randn(B, 16, 768)}
    mask = torch.ones(B, 16, dtype=torch.long)
    mask[0, 6:] = 0                                  # 人为造一点 padding

    print("{:12s} {:>10s} {:>12s}  输出键".format("fusion", "参数量", "logits 形状"))
    for name in FUSION_CHOICES:
        m = build_fusion(name)
        m.eval()
        with torch.no_grad():
            out = m(image_feat, text_feat, mask)
        extra = [k for k in out if k != "logits"]
        print("{:12s} {:>10d} {:>12s}  {}".format(
            name, sum(p.numel() for p in m.parameters()),
            str(tuple(out["logits"].shape)), extra))
        if "logits" in out and torch.isnan(out["logits"]).any():
            print("   !! 出现 NaN")
