"""The causal student: about 1.9 million parameters, seeing only the current and previous frames.

The hidden state is one tensor of 16 channels at the input resolution: the previous low-res
frame (3 channels) and 13 channels of features. Keeping the previous frame inside the state is
what lets the exported model have exactly two inputs, ``frame`` and ``state``.

Per frame:
1. the flow head estimates motion from the current to the previous frame, with a confidence map;
2. the previous features are warped by that flow and multiplied by the confidence (the gate),
   and if the mean confidence is below ``reset_threshold`` they are dropped entirely;
3. the current frame's features and the gated state go through residual blocks and a x4
   pixel-shuffle head, added to a bicubic upsampling of the frame.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.flow import FlowHead, warp

STATE_CHANNELS = 16
FEAT_CHANNELS = STATE_CHANNELS - 3


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(x)))


class Student(nn.Module):
    def __init__(self, ch: int = 80, blocks: int = 14, scale: int = 4, reset_threshold: float = 0.35):
        super().__init__()
        self.scale, self.reset_threshold = scale, reset_threshold
        self.flow = FlowHead()
        self.head = nn.Conv2d(3 + FEAT_CHANNELS, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.to_state = nn.Conv2d(ch, FEAT_CHANNELS, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.1),
            nn.Conv2d(ch, 3 * 4, 3, padding=1), nn.PixelShuffle(2),
        )

    def init_state(self, frame: torch.Tensor) -> torch.Tensor:
        n, _, h, w = frame.shape
        return torch.cat([frame, frame.new_zeros(n, FEAT_CHANNELS, h, w)], dim=1)

    def forward(self, frame: torch.Tensor, state: torch.Tensor | None = None,
                return_confidence: bool = False):
        if state is None:
            state = self.init_state(frame)
        prev_frame, prev_feat = state[:, :3], state[:, 3:]
        flow, conf = self.flow(frame, prev_frame)
        gate = conf * (conf.mean(dim=(1, 2, 3), keepdim=True) >= self.reset_threshold).to(conf.dtype)
        carried = warp(prev_feat, flow) * gate
        x = self.body(F.relu(self.head(torch.cat([frame, carried], dim=1))))
        out = self.up(x) + F.interpolate(frame, scale_factor=self.scale, mode="bicubic", align_corners=False)
        new_state = torch.cat([frame, torch.tanh(self.to_state(x))], dim=1)
        if return_confidence:
            return out.clamp(0, 1), new_state, conf
        return out.clamp(0, 1), new_state

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kw) -> Student:
        """A local ``.pt`` state dict, or a Hugging Face repo holding ``student.pt``."""
        path = Path(name_or_path)
        if not path.exists():
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(name_or_path, "student.pt"))
        model = cls(**kw)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
