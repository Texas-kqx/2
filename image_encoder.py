"""image_encoder.py —— 图像编码器：把一张图变成一个向量（或一组 token）。

教学要点
--------
1. ResNet 的最后两层 avgpool + fc 被去掉后，输出的是 [B, C, h, w] 特征图
   （224 输入 -> [B, 512, 7, 7]）。它同时支持两种用法：
      * 池化成一个全局向量 [B, 512]        -> Early / Late 融合用；
      * 展平成 49 个 token [B, 49, 512]    -> Attention 融合用
        （把图像的每个空间位置当成一个"视觉词"，才能和文本 token 做注意力交互）
2. freeze=True 时主干不训练：本实验只需训练融合层 + 分类头，
   CPU 上就能在几分钟内跑完一轮，非常适合课堂。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# backbone 名称 -> (构造函数, ImageNet 预训练权重枚举名)
_RESNET_ZOO = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet34": ("resnet34", "ResNet34_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
}


def _build_resnet(backbone: str, pretrained: bool):
    """兼容新旧 torchvision 的预训练参数写法。"""
    if backbone not in _RESNET_ZOO:
        raise ValueError(f"目前支持 {list(_RESNET_ZOO)}，收到：{backbone}")
    ctor_name, weights_name = _RESNET_ZOO[backbone]
    ctor = getattr(models, ctor_name)
    try:  # torchvision >= 0.13
        weights_cls = getattr(models, weights_name, None)
        weights = None
        if pretrained and weights_cls is not None:
            weights = weights_cls.IMAGENET1K_V1
        return ctor(weights=weights)
    except TypeError:  # 老版本 torchvision
        return ctor(pretrained=pretrained)


class ImageEncoder(nn.Module):
    """ResNet 图像编码器。

    Args:
        backbone:   resnet18 / resnet34 / resnet50
        pretrained: 是否加载 ImageNet 预训练权重（强烈建议 True）
        freeze:     是否冻结主干（只训练后面的投影层与融合层）
        out_dim:    输出维度。None 表示保持主干原生维度（resnet18 -> 512）
        dropout:    投影层 dropout
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = False,
                 out_dim: int = None, dropout: float = 0.0):
        super().__init__()
        net = _build_resnet(backbone, pretrained)
        # 去掉最后的 avgpool 与 fc，只保留到 layer4 -> [B, C, h, w]
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.backbone = backbone
        self.feature_dim = net.fc.in_features          # resnet18 -> 512, resnet50 -> 2048
        self.out_dim = self.feature_dim if out_dim is None else out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.feature_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.freeze = freeze
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "ImageEncoder":
        """冻结时把主干固定在 eval 模式：BatchNorm 的 running stats 不再更新，
        否则即使 requires_grad=False，BN 仍会在训练中被改写，导致"冻结"名不副实。"""
        super().train(mode)
        if self.freeze:
            self.body.eval()
        return self

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]，已经过 ImageNet 标准化
            return_tokens: True -> [B, h*w, out_dim]（空间 token）
                           False -> [B, out_dim]（全局特征）
        """
        fmap = self.body(x)                                  # [B, C, h, w]
        if return_tokens:
            tokens = fmap.flatten(2).transpose(1, 2)         # [B, h*w, C]
            return self.proj(tokens)                         # Linear 作用在最后一维
        pooled = self.pool(fmap).flatten(1)                  # [B, C]
        return self.proj(pooled)

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone}, feature_dim={self.feature_dim}, "
                f"out_dim={self.out_dim}, frozen={self.freeze}")
