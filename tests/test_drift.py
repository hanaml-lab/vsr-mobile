import math

import pytest
import torch
import yaml

from train.data import SyntheticClips
from train.distill import clip_loss, loss_weights, train
from train.drift import consistency_matching, contraction_penalty, drift_bound, gradient_loss, jacobian_spectral_norm
from train.student import Student


def test_drift_bound_matches_the_proposition():
    assert drift_bound(eps=0.01, gamma=0.9) == pytest.approx(0.1)  # sup over t: eps / (1 - gamma)
    assert drift_bound(eps=0.01, gamma=0.9, t=1) == pytest.approx(0.01)
    assert drift_bound(eps=0.01, gamma=0.9, t=2) == pytest.approx(0.019)  # 0.01 + 0.9 * 0.01
    assert drift_bound(eps=0.0, gamma=0.5, delta0=1.0, t=3) == pytest.approx(0.125)
    assert drift_bound(eps=0.01, gamma=1.0, t=10) == pytest.approx(0.1)  # linear growth at gamma = 1
    assert math.isinf(drift_bound(eps=0.01, gamma=1.1))


def test_power_iteration_finds_the_largest_singular_value_per_clip():
    scale = torch.ones(2, 4, 3, 3)
    scale[0, 1, 2, 2] = 2.0  # clip 0: the map stretches one direction by 2
    scale[1] *= 0.5  # clip 1: it shrinks everything by half
    sigma, _ = jacobian_spectral_norm(lambda h: h * scale, torch.rand(2, 4, 3, 3), iters=30)
    assert sigma.tolist() == pytest.approx([2.0, 0.5], rel=1e-2)


def test_contraction_penalty_is_the_mean_squared_excess_over_gamma():
    assert float(contraction_penalty(torch.tensor([0.8, 1.1]), gamma=0.9)) == pytest.approx(0.02)


def test_consistency_matching_accepts_the_teachers_motion_and_rejects_temporal_blur():
    teacher = SyntheticClips(n=1, frames=4, size=32)[0]["hr"][None]  # (1, T, 3, H, W), a moving texture
    assert float(consistency_matching(teacher, teacher)) == 0.0
    assert float(consistency_matching(teacher + 0.1, teacher)) == pytest.approx(0.0, abs=1e-6)
    blurred = teacher.mean(dim=1, keepdim=True).expand_as(teacher)  # no flicker, and no motion either
    assert float(consistency_matching(blurred, teacher)) > 0.01


def test_gradient_loss_compares_texture_not_brightness():
    flat = torch.zeros(1, 1, 4, 4)
    edge = flat.clone()
    edge[..., 2:] = 1.0  # one vertical edge: one of three horizontal differences per row is 1
    assert float(gradient_loss(flat + 0.5, flat)) == 0.0
    assert float(gradient_loss(edge, flat)) == pytest.approx(1 / 3)


def test_the_contraction_term_reaches_the_student_weights():
    torch.manual_seed(0)
    model = Student(ch=8, blocks=1)
    batch = next(iter(torch.utils.data.DataLoader(SyntheticClips(n=2, frames=3, size=32), batch_size=2)))
    _, parts = clip_loss(model, batch["lr"], batch["hr"], batch["teacher"], w_temporal=0.0,
                         w_consistency=0.5, w_texture=0.5, w_contraction=1.0, gamma=0.0)
    assert {"consistency", "texture", "contraction", "state_norm"} <= parts.keys()
    assert parts["contraction"] > 0
    only, _ = clip_loss(model, batch["lr"], batch["hr"], batch["teacher"], w_teacher=0.0, w_gt=0.0,
                        w_temporal=0.0, w_contraction=1.0, gamma=0.0)
    model.zero_grad()
    only.backward()
    assert model.to_state.weight.grad.abs().sum() > 0


def test_the_drift_config_trains():
    torch.manual_seed(0)
    cfg = yaml.safe_load(open("configs/student-x4-drift.yaml"))
    cfg["model"].update(channels=16, blocks=2)
    cfg["data"] = {"source": "synthetic", "clips": 8, "frames": 3, "crop": 32}
    cfg["train"].update(steps=20, batch_size=4, workers=0, log_every=1000, lr=2e-3)
    batch = next(iter(torch.utils.data.DataLoader(SyntheticClips(n=4, frames=3, size=32), batch_size=4)))
    kwargs = loss_weights(cfg["loss"])
    assert kwargs["w_contraction"] > 0 and kwargs["w_consistency"] > 0 and kwargs["w_temporal"] == 0
    before, _ = clip_loss(Student(ch=16, blocks=2), batch["lr"], batch["hr"], batch["teacher"], **kwargs)
    after, _ = clip_loss(train(cfg), batch["lr"], batch["hr"], batch["teacher"], **kwargs)
    assert float(after) < float(before)
