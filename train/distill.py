"""Distillation from BasicVSR++ with a temporal-consistency loss.

    python -m train.distill --config configs/student-x4.yaml

Teacher outputs are precomputed once (BasicVSR++ is bidirectional and slow) and read from
``teacher_root``. The loss is Charbonnier distance to the teacher, a smaller weight of
Charbonnier distance to ground truth, and a temporal term: consecutive outputs, the earlier
one warped onto the later by the flow the student estimated (upsampled), should agree where
the flow is confident. That term is what keeps flicker down.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from train.data import build_dataset
from train.flow import warp
from train.student import Student


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((a - b) ** 2 + eps**2).mean()


def clip_loss(model: Student, lr: torch.Tensor, hr: torch.Tensor, teacher: torch.Tensor,
              w_teacher: float = 1.0, w_gt: float = 0.25, w_temporal: float = 0.5) -> tuple[torch.Tensor, dict]:
    """lr: (N, T, 3, h, w); hr and teacher: (N, T, 3, H, W)."""
    state = None
    outs, confs, flows = [], [], []
    for t in range(lr.shape[1]):
        frame = lr[:, t]
        prev = state[:, :3] if state is not None else frame
        out, state, conf = model(frame, state, return_confidence=True)
        flow, _ = model.flow(frame, prev)
        outs.append(out)
        confs.append(conf)
        flows.append(flow)
    out = torch.stack(outs, dim=1)
    distill = charbonnier(out, teacher)
    gt = charbonnier(out, hr)
    temporal = out.new_zeros(())
    s = model.scale
    for t in range(1, len(outs)):
        flow_hr = F.interpolate(flows[t], scale_factor=s, mode="bilinear", align_corners=False) * s
        conf_hr = F.interpolate(confs[t], scale_factor=s, mode="bilinear", align_corners=False).detach()
        warped_prev = warp(outs[t - 1], flow_hr)
        temporal = temporal + (conf_hr * (outs[t] - warped_prev).abs()).mean()
    temporal = temporal / max(len(outs) - 1, 1)
    loss = w_teacher * distill + w_gt * gt + w_temporal * temporal
    return loss, {"distill": float(distill), "gt": float(gt), "temporal": float(temporal)}


def train(cfg: dict, steps: int | None = None, device: str = "cpu") -> Student:
    m = cfg["model"]
    model = Student(ch=m["channels"], blocks=m["blocks"], scale=m["scale"],
                    reset_threshold=m["reset_threshold"]).to(device)
    data = build_dataset(cfg["data"], scale=m["scale"])
    loader = torch.utils.data.DataLoader(data, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                         num_workers=cfg["train"].get("workers", 0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], betas=(0.9, 0.99))
    total = steps or cfg["train"]["steps"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    w = cfg["loss"]
    step = 0
    while step < total:
        for batch in loader:
            loss, parts = clip_loss(model, batch["lr"].to(device), batch["hr"].to(device),
                                    batch["teacher"].to(device), w["teacher"], w["gt"], w["temporal"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % cfg["train"].get("log_every", 100) == 0:
                print(f"step {step} loss {float(loss):.4f} {parts}", flush=True)
            if step >= total:
                break
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/student.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    model = train(cfg, device=args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
