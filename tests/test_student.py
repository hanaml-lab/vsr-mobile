import numpy as np
import pytest
import torch

from train.data import SyntheticClips, degrade
from train.distill import clip_loss, train
from train.flow import warp
from train.student import Student, count_parameters


def test_parameter_budget_and_shapes():
    model = Student()
    assert 1.7e6 < count_parameters(model) < 2.1e6
    out, state = model(torch.rand(1, 3, 24, 40))
    assert out.shape == (1, 3, 96, 160) and state.shape == (1, 16, 24, 40)
    assert torch.equal(state[:, :3], state[:, :3].clamp(0, 1))


def test_warp_by_integer_flow_shifts_pixels():
    x = torch.arange(25, dtype=torch.float32).view(1, 1, 5, 5)
    flow = torch.zeros(1, 2, 5, 5)
    flow[:, 0] = 1.0  # sample one pixel to the right
    out = warp(x, flow)
    assert torch.allclose(out[0, 0, :, :4], x[0, 0, :, 1:])


def test_gate_resets_state_at_a_cut():
    torch.manual_seed(0)
    model = Student(reset_threshold=1.01).eval()  # a threshold no confidence reaches: always reset
    frames = torch.rand(2, 1, 3, 16, 16)
    with torch.no_grad():
        _, s1 = model(frames[0])
        out_reset, _ = model(frames[1], s1)
        no_memory = torch.cat([s1[:, :3], torch.zeros_like(s1[:, 3:])], dim=1)  # same previous frame, no features
        out_fresh, _ = model(frames[1], no_memory)
    assert torch.allclose(out_reset, out_fresh, atol=1e-5)
    keep = Student(reset_threshold=0.0).eval()
    keep.load_state_dict(model.state_dict())
    with torch.no_grad():
        _, s1 = keep(frames[0])
        s1[:, 3:] = 1.0  # a strong memory
        assert not torch.allclose(keep(frames[1], s1)[0], keep(frames[1], no_memory)[0])


def test_degrade_shapes():
    hr = torch.rand(2, 3, 64, 64)
    assert degrade(hr, 4, blur_sigma=1.0, noise=0.01).shape == (2, 3, 16, 16)


def test_a_few_steps_reduce_the_loss():
    torch.manual_seed(0)
    cfg = {"model": {"channels": 16, "blocks": 2, "scale": 4, "reset_threshold": 0.35},
           "data": {"source": "synthetic", "clips": 8, "frames": 3, "crop": 64},
           "train": {"batch_size": 4, "lr": 2e-3, "steps": 30, "log_every": 1000},
           "loss": {"teacher": 1.0, "gt": 0.25, "temporal": 0.5}}
    batch = next(iter(torch.utils.data.DataLoader(SyntheticClips(n=4, frames=3, size=16), batch_size=4)))
    fresh = Student(ch=16, blocks=2)
    before, _ = clip_loss(fresh, batch["lr"], batch["hr"], batch["teacher"])
    trained = train(cfg)
    after, _ = clip_loss(trained, batch["lr"], batch["hr"], batch["teacher"])
    assert float(after) < float(before)


def test_onnx_export_runs_in_onnxruntime(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from export.onnx import export

    model = Student(ch=16, blocks=2).eval()
    path = export(model, tmp_path / "vsr.onnx", height=18, width=32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    frame = np.random.rand(1, 3, 18, 32).astype(np.float32)
    up, state = sess.run(None, {"frame": frame, "state": np.zeros((1, 16, 18, 32), np.float32)})
    with torch.no_grad():
        ref, ref_state = model(torch.from_numpy(frame))
    assert up.shape == (1, 3, 72, 128)
    assert np.allclose(up, ref.numpy(), atol=1e-4) and np.allclose(state, ref_state.numpy(), atol=1e-4)
