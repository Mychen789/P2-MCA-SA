import os

import torch
import torch.nn as nn

from nmv.modules.attention import EMA


class _CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021). Channel-preserving."""

    def __init__(self, c, reduction=16):
        super().__init__()
        mid = max(8, c // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c, mid, 1, 1, 0)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, c, 1, 1, 0)
        self.conv_w = nn.Conv2d(mid, c, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


class MCA(nn.Module):
    """Multi-branch Cross Attention (本工作新提出).

    Fuses EMA (multi-group cross-spatial) with CoordAtt (directional coord-aware)
    via gated concat + learnable weighted sum. Channel-preserving.

    Yaml row: `[-1, 1, MCA, [c, factor]]`.
    """

    def __init__(self, c1, c2, factor=4):
        super().__init__()
        self.ema = EMA(c1, c2, factor=factor)
        self.ca = _CoordAtt(c2)
        self.gate = nn.Conv2d(c2 * 2, c2, 1, 1, 0)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(c2, c2, 1, 1, 0)
        # 组件消融开关 (NMV_MCA_MODE=ema|ca|full)：在同一 yolov8m-p2-mca.yaml 图上隔离
        # EMA-only / CoordAtt-only / 完整门控融合，反驳"MCA 只是两个注意力堆叠"。默认 full。
        self.mode = os.environ.get("NMV_MCA_MODE", "full").lower()

    def forward(self, x):
        # getattr 兜底：2026-05-28 之前序列化的 best.pt instance 没有 mode 字段。
        mode = getattr(self, "mode", "full")
        if mode == "ema":
            return self.ema(x)
        if mode == "ca":
            return self.ca(x)
        e = self.ema(x)
        c = self.ca(x)
        g = self.gate(torch.cat([e, c], dim=1)).sigmoid()
        y = g * (self.alpha * e + self.beta * c)
        return self.fuse(y)
