"""Training clips: REDS-style folders of frames, degraded to low resolution on the fly, and a
synthetic moving-pattern source for smoke tests."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def degrade(hr: torch.Tensor, scale: int = 4, blur_sigma: float = 0.0, noise: float = 0.0,
            jpeg_quality: int | None = None) -> torch.Tensor:
    """(T, 3, H, W) in [0, 1] -> (T, 3, H/scale, W/scale): blur, bicubic downsample, noise, JPEG."""
    x = hr
    if blur_sigma > 0:
        k = int(blur_sigma * 3) * 2 + 1
        ax = torch.arange(k, dtype=x.dtype) - k // 2
        g = torch.exp(-ax**2 / (2 * blur_sigma**2))
        g = (g / g.sum()).to(x.device)
        x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
        x = F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    lr = F.interpolate(x, scale_factor=1 / scale, mode="bicubic", align_corners=False, antialias=True)
    if noise > 0:
        lr = lr + noise * torch.randn_like(lr)
    lr = lr.clamp(0, 1)
    if jpeg_quality is not None:
        lr = torch.stack([_jpeg(f, jpeg_quality) for f in lr])
    return lr


def _jpeg(frame: torch.Tensor, quality: int) -> torch.Tensor:
    from PIL import Image

    arr = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(out).permute(2, 0, 1)


class REDSClips(torch.utils.data.Dataset):
    """``root/<clip>/<frame>.png`` (REDS train_sharp layout). Each item is ``frames`` consecutive
    frames cropped to ``crop`` high-res pixels, with teacher outputs from ``teacher_root`` when given."""

    def __init__(self, root: str | Path, frames: int = 7, crop: int = 256, scale: int = 4,
                 teacher_root: str | Path | None = None, seed: int = 0):
        self.clips = sorted(p for p in Path(root).iterdir() if p.is_dir())
        self.frames, self.crop, self.scale = frames, crop, scale
        self.teacher_root = Path(teacher_root) if teacher_root else None
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clips) * 10

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        from PIL import Image

        return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)

    def __getitem__(self, i: int) -> dict:
        clip = self.clips[i % len(self.clips)]
        names = sorted(clip.glob("*.png"))
        start = self.rng.randrange(0, len(names) - self.frames + 1)
        hr = torch.stack([self._load(p) for p in names[start:start + self.frames]])
        _, _, h, w = hr.shape
        y, x = self.rng.randrange(0, h - self.crop + 1), self.rng.randrange(0, w - self.crop + 1)
        hr = hr[..., y:y + self.crop, x:x + self.crop]
        item = {"lr": degrade(hr, self.scale, blur_sigma=self.rng.uniform(0, 1.2), noise=self.rng.uniform(0, 0.02),
                              jpeg_quality=self.rng.choice([None, 60, 75, 90])), "hr": hr}
        if self.teacher_root:
            window = names[start:start + self.frames]
            t = torch.stack([self._load(self.teacher_root / clip.name / p.name) for p in window])
            item["teacher"] = t[..., y:y + self.crop, x:x + self.crop]
        return item


class SyntheticClips(torch.utils.data.Dataset):
    """Textured patterns translating at a constant velocity, with an optional hard cut half-way.
    Enough to check that training reduces loss and that the gate fires at a cut."""

    def __init__(self, n: int = 16, frames: int = 6, size: int = 64, scale: int = 4, cut: bool = False, seed: int = 0):
        self.n, self.frames, self.size, self.scale, self.cut, self.seed = n, frames, size, scale, cut, seed

    def __len__(self) -> int:
        return self.n

    def _texture(self, g: torch.Generator, big: int) -> torch.Tensor:
        base = torch.rand(3, big // 8, big // 8, generator=g)
        return F.interpolate(base[None], size=(big, big), mode="bicubic", align_corners=False)[0].clamp(0, 1)

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 1000 + i)
        big = self.size * 2
        tex, other = self._texture(g, big), self._texture(g, big)
        vy, vx = (int(v) for v in torch.randint(-3, 4, (2,), generator=g))
        frames = []
        for t in range(self.frames):
            src = other if self.cut and t >= self.frames // 2 else tex
            oy, ox = self.size // 2 + vy * t, self.size // 2 + vx * t
            frames.append(src[:, oy:oy + self.size, ox:ox + self.size])
        hr = torch.stack(frames)
        return {"lr": degrade(hr, self.scale), "hr": hr, "teacher": hr}
