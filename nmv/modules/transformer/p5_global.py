"""P5 Transformer global branch with gated residual fusion to CNN P5 features.

Diagnosis (2026-05-22): the P2 high-resolution head introduced for small objects
collapses mAP_large by ~20pp (E01 0.509 -> E02 0.303) even before any attention
is added. Scale-aware label assignment (E16', HI_RATIO=32) recovers part of it
(large 0.247 -> 0.390) but never matches the pre-P2 baseline 0.509. The residual
gap is attributed to the P5 feature pathway itself: at stride=32, the CNN-only
backbone has limited receptive field and global context on the largest objects
that span sizable portions of the 960x960 input.

Fix: attach a lightweight Transformer encoder on top of the P5/32 feature map
to inject global self-attention. The transformer branch is gated-residually fused
with the CNN P5 features so it can supplement, not replace, the CNN path. The
projection-out is zero-initialised so at start of training the transformer
branch is exactly identity (output == CNN input), letting the gate learn to open.

Yaml row: `[-1, 1, P5Transformer, [c]]` (defaults dim=192, n_layers=2, n_heads=4)
         or `[-1, 1, P5Transformer, [c, dim, n_layers, n_heads]]` for explicit config.

Memory budget (imgsz=960, stride=32 -> 30x30=900 tokens, dim=192, 2 layers, 4 heads):
- params: ~0.3M
- GFLOPs: ~4-5
- VRAM @ batch=4: <300MB extra
"""
import os

import torch
import torch.nn as nn


class P5Transformer(nn.Module):

    def __init__(self, c1, c2, dim=192, n_layers=2, n_heads=4, ffn_mult=2, max_tokens=2048):
        super().__init__()
        assert c1 == c2, f"P5Transformer is channel-preserving; got c1={c1}, c2={c2}"
        self.dim = dim
        self.max_tokens = max_tokens

        self.proj_in = nn.Conv2d(c1, dim, 1, 1, 0)
        self.proj_out = nn.Conv2d(dim, c2, 1, 1, 0)
        # Zero-init proj_out so the transformer branch starts as identity.
        # Gate must learn to open; CNN path is untouched at epoch 0.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * ffn_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability on small data
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Learnable position embedding sized for up to max_tokens (default covers
        # imgsz up to ~1440 at stride 32). Trained from scratch with the model.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Spatial gate on concatenated [cnn, trans_back] -> per-position sigmoid weight.
        self.gate = nn.Conv2d(c2 * 2, c2, 1, 1, 0)

        # Ablation switch (NMV_P5TRANS_MODE = full | trans_only | cnn_only):
        #   full       — gated residual fusion (default, used in E17 main).
        #   trans_only — ungated residual (output = cnn + trans_proj_out).
        #   cnn_only   — disable transformer branch entirely (returns input).
        self.mode = os.environ.get("NMV_P5TRANS_MODE", "full").lower()

    def forward(self, x):
        mode = getattr(self, "mode", "full")
        if mode == "cnn_only":
            return x

        b, c, h, w = x.shape
        n_tokens = h * w
        if n_tokens > self.max_tokens:
            return x

        z = self.proj_in(x)
        z = z.flatten(2).transpose(1, 2)
        z = z + self.pos_embed[:, :n_tokens, :]
        z = self.encoder(z)
        z = z.transpose(1, 2).reshape(b, self.dim, h, w)
        trans = self.proj_out(z)

        if mode == "trans_only":
            return x + trans

        g = self.gate(torch.cat([x, trans], dim=1)).sigmoid()
        return x + g * trans
