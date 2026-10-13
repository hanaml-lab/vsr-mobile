"""Distillation from BasicVSR++ with a temporal-consistency loss.

    python -m train.distill --config configs/student-x4.yaml
    python -m train.distill --config configs/student-x4-drift.yaml   # with DRIFT's terms

Teacher outputs are precomputed once (BasicVSR++ is bidirectional and slow) and read from
``teacher_root``. The loss is Charbonnier distance to the teacher, a smaller weight of
Charbonnier distance to ground truth, and a temporal term: consecutive outputs, the earlier
one warped onto the later by the flow the student estimated (upsampled), should agree where
the flow is confident. That term is what keeps flicker down.

The DRIFT config replaces the temporal term with consistency matching (agree with the teacher's
temporal residual rather than suppress all change), adds a gradient-domain texture term against
the teacher, and penalises a state-Jacobian norm above gamma (see train/drift.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from train.data import build_dataset
from train.drift import consistency_matching, contraction_penalty, gradient_loss, jacobian_spectral_norm
from train.flow import warp
from train.student import Student


def charbonnier(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((a - b) ** 2 + eps**2).mean()


def clip_loss(model: Student, lr: torch.Tensor, hr: torch.Tensor, teacher: torch.Tensor,
              w_teacher: float = 1.0, w_gt: float = 0.25, w_temporal: float = 0.5, w_consistency: float = 0.0,
              w_texture: float = 0.0, w_contraction: float = 0.0, gamma: float = 0.9) -> tuple[torch.Tensor, dict]:
    """lr: (N, T, 3, h, w); hr and teacher: (N, T, 3, H, W). Terms with a zero weight are not computed."""
    state = None
    outs, confs, flows, states = [], [], [], []
    for t in range(lr.shape[1]):
        frame = lr[:, t]
        prev = state[:, :3] if state is not None else frame
        states.append(state)
        out, state, conf = model(frame, state, return_confidence=True)
        flow, _ = model.flow(frame, prev)
        outs.append(out)
        confs.append(conf)
        flows.append(flow)
    out = torch.stack(outs, dim=1)
    s = model.scale
    flows_hr = [F.interpolate(f, scale_factor=s, mode="bilinear", align_corners=False) * s for f in flows]
    loss = w_teacher * charbonnier(out, teacher) + w_gt * charbonnier(out, hr)
    parts = {"distill": float(charbonnier(out, teacher)), "gt": float(charbonnier(out, hr))}
    if w_temporal:
        temporal = out.new_zeros(())
        for t in range(1, len(outs)):
            conf_hr = F.interpolate(confs[t], scale_factor=s, mode="bilinear", align_corners=False).detach()
            temporal = temporal + (conf_hr * (outs[t] - warp(outs[t - 1], flows_hr[t])).abs()).mean()
        temporal = temporal / max(len(outs) - 1, 1)
        loss, parts["temporal"] = loss + w_temporal * temporal, float(temporal)
    if w_consistency:
        consistency = consistency_matching(out, teacher, torch.stack(flows_hr, dim=1).detach())
        loss, parts["consistency"] = loss + w_consistency * consistency, float(consistency)
    if w_texture:
        texture = gradient_loss(out.flatten(0, 1), teacher.flatten(0, 1))
        loss, parts["texture"] = loss + w_texture * texture, float(texture)
    if w_contraction and len(outs) > 1:
        last = lr[:, -1]
        sigma, _ = jacobian_spectral_norm(lambda h: model(last, h)[1], states[-1], iters=2)
        contraction = contraction_penalty(sigma, gamma)
        loss = loss + w_contraction * contraction
        parts["contraction"], parts["state_norm"] = float(contraction), float(sigma.mean())
    return loss, parts


def loss_weights(loss_cfg: dict) -> dict:
    """The config's ``loss`` section as clip_loss keyword arguments; absent terms are off."""
    names = ("teacher", "gt", "temporal", "consistency", "texture", "contraction")
    return {f"w_{k}": loss_cfg.get(k, 0.0) for k in names} | {"gamma": loss_cfg.get("gamma", 0.9)}


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
                                    batch["teacher"].to(device), **loss_weights(w))
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
