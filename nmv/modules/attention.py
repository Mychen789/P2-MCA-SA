import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-scale Attention (Ouyang et al., ICASSP 2023).

    Drop-in attention block for YOLO necks/backbones. Channel count is preserved.
    The yaml row is `[-1, 1, EMA, [c, factor]]`; parse_model auto-injects c1, c2.
    """

    def __init__(self, c1, c2, factor=8):
        super().__init__()
        self.groups = factor
        if c2 % factor != 0:
            self.groups = max(g for g in (8, 4, 2, 1) if c2 % g == 0)
        gc = c2 // self.groups
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(gc, gc)
        self.conv1x1 = nn.Conv2d(gc, gc, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(gc, gc, 3, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        gx = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(gx)
        x_w = self.pool_w(gx).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(gx * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(gx)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (gx * weights.sigmoid()).reshape(b, c, h, w)


class SimAM(nn.Module):
    """SimAM: Parameter-free attention (Yang et al., ICML 2021).

    Zero-parameter, no extra weight memory. Same input/output channels.
    """

    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(v)
