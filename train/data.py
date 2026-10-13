"""Training clips: REDS-style folders of frames, degraded to low resolution on the fly, and a
synthetic moving-pattern source for smoke tests.

The degradation mimics a streaming pipeline: blur, one of several downscaling kernels, noise, and
then either JPEG on each frame or a real video codec (H.264 or AV1 through ffmpeg) on the whole
clip, so the student sees the blocking and ringing it will meet in decoded video.
"""

from __future__ import annotations

import functools
import io
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KERNELS = ("bicubic", "bilinear", "area")
ENCODERS = {"h264": ("libx264", ["-preset", "veryfast"]), "av1": ("libaom-av1", ["-cpu-used", "8", "-row-mt", "1"])}


@functools.cache
def codec_available(codec: str) -> bool:
    """Whether ffmpeg is on the PATH and has the encoder used for ``codec`` ("h264" or "av1")."""
    if codec not in ENCODERS or shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return f" {ENCODERS[codec][0]} " in out


def compress(clip: torch.Tensor, codec: str, crf: int) -> torch.Tensor:
    """Encode (T, 3, h, w) frames in [0, 1] as one H.264 or AV1 stream at ``crf`` and decode them again.

    Frames go through YUV 4:2:0 as in real streams, so chroma is subsampled too. h and w must be even.
    """
    encoder, opts = ENCODERS[codec]
    t, _, h, w = clip.shape
    raw = (clip.clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
    size = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25"]
    encoded = subprocess.run(["ffmpeg", "-v", "error", *size, "-i", "-", "-c:v", encoder, *opts, "-crf", str(crf),
                              "-pix_fmt", "yuv420p", "-f", "nut", "-"], input=raw, capture_output=True, check=True)
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             input=encoded.stdout, capture_output=True, check=True).stdout
    frames = np.frombuffer(decoded, np.uint8)[: t * h * w * 3].reshape(t, h, w, 3)
    return torch.from_numpy(frames.astype(np.float32) / 255).permute(0, 3, 1, 2).contiguous()


def degrade(hr: torch.Tensor, scale: int = 4, blur_sigma: float = 0.0, noise: float = 0.0,
            jpeg_quality: int | None = None, kernel: str = "bicubic", codec: str | None = None,
            crf: int = 28) -> torch.Tensor:
    """(T, 3, H, W) in [0, 1] -> (T, 3, H/scale, W/scale): blur, downsample with ``kernel``, noise,
    then JPEG per frame (``jpeg_quality``) or a video codec over the clip (``codec`` at ``crf``)."""
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, not {kernel!r}")
    x = hr
    if blur_sigma > 0:
        k = int(blur_sigma * 3) * 2 + 1
        ax = torch.arange(k, dtype=x.dtype) - k // 2
        g = torch.exp(-ax**2 / (2 * blur_sigma**2))
        g = (g / g.sum()).to(x.device)
        x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
        x = F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    if kernel == "area":
        lr = F.avg_pool2d(x, scale)
    else:
        lr = F.interpolate(x, scale_factor=1 / scale, mode=kernel, align_corners=False, antialias=True)
    if noise > 0:
        lr = lr + noise * torch.randn_like(lr)
    lr = lr.clamp(0, 1)
    if jpeg_quality is not None:
        lr = torch.stack([_jpeg(f, jpeg_quality) for f in lr])
    if codec is not None:
        lr = compress(lr, codec, crf)
    return lr


def _jpeg(frame: torch.Tensor, quality: int) -> torch.Tensor:
    from PIL import Image

    arr = (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(out).permute(2, 0, 1)


def random_degradation(rng: random.Random) -> dict:
    """One draw of the streaming-like degradation used for training clips: keyword arguments for degrade."""
    kw = {"blur_sigma": rng.uniform(0, 1.2), "noise": rng.uniform(0, 0.02), "kernel": rng.choice(KERNELS)}
    choice = rng.choice([None, "jpeg", *(c for c in ENCODERS if codec_available(c))])
    if choice == "jpeg":
        kw["jpeg_quality"] = rng.choice([60, 75, 90])
    elif choice is not None:
        kw["codec"], kw["crf"] = choice, rng.randint(23, 40)
    return kw


def _load(path: Path) -> torch.Tensor:
    from PIL import Image

    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)


class _FrameClips(torch.utils.data.Dataset):
    """Clips given as folders of frame files. Each item is ``frames`` consecutive frames cropped to
    ``crop`` high-res pixels and degraded with a fresh random draw, with the teacher's outputs for the
    same frames and crop when ``teacher_root`` holds them under the same relative paths."""

    def __init__(self, root: Path, clips: list[Path], frames: int, crop: int, scale: int,
                 teacher_root: str | Path | None, seed: int, windows_per_clip: int):
        self.root, self.clips = root, clips
        self.frames, self.crop, self.scale, self.windows = frames, crop, scale, windows_per_clip
        self.teacher_root = Path(teacher_root) if teacher_root else None
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clips) * self.windows

    def __getitem__(self, i: int) -> dict:
        clip = self.clips[i % len(self.clips)]
        names = sorted(clip.glob("*.png"))
        start = self.rng.randrange(0, len(names) - self.frames + 1)
        window = names[start:start + self.frames]
        hr = torch.stack([_load(p) for p in window])
        _, _, h, w = hr.shape
        y, x = self.rng.randrange(0, h - self.crop + 1), self.rng.randrange(0, w - self.crop + 1)
        hr = hr[..., y:y + self.crop, x:x + self.crop]
        item = {"lr": degrade(hr, self.scale, **random_degradation(self.rng)), "hr": hr}
        if self.teacher_root:
            rel = clip.relative_to(self.root)
            t = torch.stack([_load(self.teacher_root / rel / p.name) for p in window])
            item["teacher"] = t[..., y:y + self.crop, x:x + self.crop]
        return item


class REDSClips(_FrameClips):
    """``root/<clip>/<frame>.png`` (REDS train_sharp layout), ten random windows per clip per epoch."""

    def __init__(self, root: str | Path, frames: int = 7, crop: int = 256, scale: int = 4,
                 teacher_root: str | Path | None = None, seed: int = 0):
        root = Path(root)
        super().__init__(root, sorted(p for p in root.iterdir() if p.is_dir()), frames, crop, scale,
                         teacher_root, seed, windows_per_clip=10)


class Vimeo90KSeptuplets(_FrameClips):
    """Vimeo-90K septuplets: ``root/sequences/<group>/<clip>/im1.png`` to ``im7.png``, the clips listed
    in ``root/sep_trainlist.txt``. Teacher outputs live at ``teacher_root/<group>/<clip>/im<k>.png``."""

    def __init__(self, root: str | Path, frames: int = 7, crop: int = 256, scale: int = 4,
                 teacher_root: str | Path | None = None, seed: int = 0, split: str = "sep_trainlist.txt"):
        root = Path(root)
        listed = [line.strip() for line in (root / split).read_text().splitlines() if line.strip()]
        super().__init__(root / "sequences", [root / "sequences" / c for c in listed], frames, crop, scale,
                         teacher_root, seed, windows_per_clip=1)


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


def build_dataset(data: dict, scale: int, seed: int = 0) -> torch.utils.data.Dataset:
    """The training set a config's ``data`` section names: ``source`` is ``synthetic``, ``reds``,
    ``vimeo90k`` or a list of the last two, which are concatenated. Each real source reads ``root`` and
    ``teacher_root`` from its own section (``data.reds``, ``data.vimeo90k``)."""
    sources = data["source"] if isinstance(data["source"], list) else [data["source"]]
    if sources == ["synthetic"]:
        return SyntheticClips(n=data.get("clips", 64), frames=data["frames"], size=data["crop"],
                              scale=scale, seed=seed)
    kinds = {"reds": REDSClips, "vimeo90k": Vimeo90KSeptuplets}
    unknown = [s for s in sources if s not in kinds]
    if unknown:
        raise ValueError(f"unknown data source(s) {unknown}; use synthetic, reds or vimeo90k")
    parts = [kinds[s](data[s]["root"], frames=data["frames"], crop=data["crop"], scale=scale,
                      teacher_root=data[s].get("teacher_root"), seed=seed) for s in sources]
    return parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
