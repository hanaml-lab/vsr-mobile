"""Training terms from DRIFT (hanaML technical report, 2025), carried over from its reference
implementation, github.com/hanaml-lab/ondevice-video-sr, for the production student.

- ``drift_bound``: the paper's Proposition 1. If the state update is a contraction with modulus
  gamma < 1 and each step adds at most eps of rounding error, the state error stays below
  gamma^t delta0 + eps (1 - gamma^t) / (1 - gamma) for every t; at gamma >= 1 it grows with t.
- ``jacobian_spectral_norm`` and ``contraction_penalty``: the regulariser that keeps the
  state-Jacobian spectral norm below gamma, estimated by power iteration without forming the Jacobian.
- ``consistency_matching``: match the teacher's flow-compensated temporal residual instead of
  suppressing temporal change, which would reward temporal blur.
- ``gradient_loss``: a texture term on image gradients, an offline stand-in for LPIPS.

The paper's gated contractive update and its long-rollout state bank are not here: they change the
architecture, and vsr-lite's weights must keep loading. See configs/student-x4-drift.yaml.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from train.flow import warp


def drift_bound(eps: float, gamma: float, delta0: float = 0.0, t: int | None = None) -> float:
    """Bound on the state error after ``t`` steps, or over all steps when ``t`` is None."""
    if gamma >= 1.0:
        if t is None:
            return float("inf")
        return gamma**t * delta0 + (eps * t if gamma == 1.0 else eps * (gamma**t - 1) / (gamma - 1))
    if t is None:
        return delta0 + eps / (1 - gamma)
    return gamma**t * delta0 + eps * (1 - gamma**t) / (1 - gamma)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (v.dim() - 1))


def jacobian_spectral_norm(fn: Callable[[torch.Tensor], torch.Tensor], h: torch.Tensor, iters: int = 2,
                           eps: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """(sigma, v): sigma estimates ||d fn / d h||_2 at ``h`` for each batch element.

    Power iteration v <- J^T J v uses a central finite-difference Jacobian-vector product (the warp's
    grid sampling has no forward-mode derivative) and an autograd vector-Jacobian product. Only the
    last ||J v|| keeps the graph, so the estimate can be penalised and its gradient reaches the weights.
    Power iteration approaches sigma from below, so the penalty is a lower bound on the violation.
    """
    h = h.detach()
    v = _unit(torch.randn_like(h))
    for _ in range(max(iters - 1, 0)):
        with torch.no_grad():
            jv = (fn(h + eps * v) - fn(h - eps * v)) / (2 * eps)
        hr = h.clone().requires_grad_(True)
        with torch.enable_grad():
            (w,) = torch.autograd.grad(fn(hr), hr, jv)
        v = _unit(w.detach())
    jv = (fn(h + eps * v) - fn(h - eps * v)) / (2 * eps)
    return jv.flatten(1).norm(dim=1), v


def contraction_penalty(sigma: torch.Tensor, gamma: float) -> torch.Tensor:
    """Mean of (sigma - gamma)_+^2."""
    return torch.relu(sigma - gamma).pow(2).mean()


def temporal_residual(clip: torch.Tensor, flow_hr: torch.Tensor | None = None) -> torch.Tensor:
    """r_t = y_t - W(y_{t-1}) for a clip (N, T, C, H, W); ``flow_hr`` (N, T, 2, H, W) warps frame t-1
    onto t (its entry at t = 0 is unused). Without a flow, r_t is the plain frame difference."""
    prev, cur = clip[:, :-1], clip[:, 1:]
    if flow_hr is None:
        return cur - prev
    return cur - warp(prev.flatten(0, 1), flow_hr[:, 1:].flatten(0, 1)).view_as(prev)


def consistency_matching(student: torch.Tensor, teacher: torch.Tensor,
                         flow_hr: torch.Tensor | None = None) -> torch.Tensor:
    """||r_t(student) - r_t(teacher)||_1: penalise only the temporal change the teacher does not have."""
    return (temporal_residual(student, flow_hr) - temporal_residual(teacher, flow_hr)).abs().mean()


def gradient_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L1 between the horizontal and vertical differences of two images: texture, not brightness."""
    return ((a[..., :, 1:] - a[..., :, :-1]) - (b[..., :, 1:] - b[..., :, :-1])).abs().mean() + \
        ((a[..., 1:, :] - a[..., :-1, :]) - (b[..., 1:, :] - b[..., :-1, :])).abs().mean()
