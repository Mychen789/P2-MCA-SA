import torch
import torch.nn as nn


class _ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k, d=1):
        super().__init__()
        p = (k - 1) // 2 * d
        self.conv = nn.Conv2d(c_in, c_out, k, 1, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ASPPLite(nn.Module):
    """Lightweight ASPP (Atrous Spatial Pyramid Pooling), 3 parallel branches.

    Used as context augmentation inside GFPN skip paths (CAGFPN).
    Yaml row: `[-1, 1, ASPPLite, [c_out]]` or `[from, 1, ASPPLite, [c_in, c_out]]`.
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_each = c2 // 3
        c_last = c2 - 2 * c_each
        self.b1 = _ConvBNAct(c1, c_each, k=1, d=1)
        self.b2 = _ConvBNAct(c1, c_each, k=3, d=3)
        self.b3 = _ConvBNAct(c1, c_last, k=3, d=5)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
