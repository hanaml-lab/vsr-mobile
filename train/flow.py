"""A small optical-flow head and the backward warp used to carry the hidden state forward.

The head is a three-level pyramid of residual flow estimators in the spirit of RAFT's update
block, cut down to fit a phone: each level refines the upsampled flow from the coarser one. It
also predicts a per-pixel confidence in [0, 1], trained to be low where warping the previous
frame does not reproduce the current one (cuts, fast motion, occlusions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``x`` (N, C, H, W) by ``flow`` (N, 2, H, W) in pixels: out[p] = x[p + flow[p]]."""
    n, _, h, w = x.shape
    if h < 2 or w < 2:  # the sampling grid divides by h - 1 and w - 1
        raise ValueError(f"warp needs at least 2 x 2 pixels, got {h} x {w}")
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    gx = (xs + flow[:, 0]) / (w - 1) * 2 - 1  # frames are at least 2 pixels on each side
    gy = (ys + flow[:, 1]) / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class _Level(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, ch, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(ch, ch, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, cur, prev, flow):
        warped = warp(prev, flow)
        out = self.net(torch.cat([cur, warped, flow], dim=1))
        return flow + out[:, :2], out[:, 2:3]


class FlowHead(nn.Module):
    def __init__(self, ch: int = 32, levels: int = 3):
        super().__init__()
        self.levels = nn.ModuleList(_Level(ch) for _ in range(levels))

    def forward(self, cur: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow from ``cur`` to ``prev`` and a confidence map, both at full resolution."""
        side = 2 ** len(self.levels)
        if min(cur.shape[-2:]) < side:
            raise ValueError(f"the {len(self.levels)}-level flow pyramid needs frames of at least {side} x {side} "
                             f"pixels, got {cur.shape[-2]} x {cur.shape[-1]}")
        pyramid = [(cur, prev)]
        for _ in range(len(self.levels) - 1):
            c, p = pyramid[-1]
            pyramid.append((F.avg_pool2d(c, 2), F.avg_pool2d(p, 2)))
        flow = torch.zeros_like(pyramid[-1][0][:, :2])
        logit = torch.zeros_like(pyramid[-1][0][:, :1])
        for level, (c, p) in zip(self.levels, reversed(pyramid), strict=True):
            if flow.shape[-2:] != c.shape[-2:]:
                flow = F.interpolate(flow, size=c.shape[-2:], mode="bilinear", align_corners=False) * 2
                logit = F.interpolate(logit, size=c.shape[-2:], mode="bilinear", align_corners=False)
            flow, delta = level(c, p, flow)
            logit = logit + delta
        return flow, torch.sigmoid(logit)
