import numpy as np
import pytest
import torch

from train.data import SyntheticClips, Vimeo90KSeptuplets, build_dataset, codec_available, degrade
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


def test_frames_too_small_for_the_flow_pyramid_are_refused():
    with pytest.raises(ValueError, match="at least 8 x 8"):
        Student(ch=16, blocks=2)(torch.rand(1, 3, 4, 4))
    with pytest.raises(ValueError, match="2 x 2"):
        warp(torch.rand(1, 1, 1, 5), torch.zeros(1, 2, 1, 5))


def test_degrade_shapes():
    hr = torch.rand(2, 3, 64, 64)
    assert degrade(hr, 4, blur_sigma=1.0, noise=0.01).shape == (2, 3, 16, 16)


def test_a_few_steps_reduce_the_loss():
    torch.manual_seed(0)
    cfg = {"model": {"channels": 16, "blocks": 2, "scale": 4, "reset_threshold": 0.35},
           "data": {"source": "synthetic", "clips": 8, "frames": 3, "crop": 32},
           "train": {"batch_size": 4, "lr": 2e-3, "steps": 30, "log_every": 1000},
           "loss": {"teacher": 1.0, "gt": 0.25, "temporal": 0.5}}
    batch = next(iter(torch.utils.data.DataLoader(SyntheticClips(n=4, frames=3, size=32), batch_size=4)))
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


def _clip(size=128, frames=4):
    return SyntheticClips(n=1, frames=frames, size=size)[0]["hr"]


@pytest.mark.parametrize("codec", ["h264", "av1"])
def test_video_codec_degradation_loses_more_detail_at_a_higher_crf(codec):
    if not codec_available(codec):
        pytest.skip(f"ffmpeg has no {codec} encoder")
    hr = _clip()
    clean = degrade(hr, 4)
    low, high = (degrade(hr, 4, codec=codec, crf=crf) for crf in (18, 45))
    assert low.shape == high.shape == clean.shape == (4, 3, 32, 32)
    err = [float(((x - clean) ** 2).mean()) for x in (low, high)]
    assert 0 < err[0] < err[1]


def test_downscaling_kernels_differ_but_keep_flat_images_flat():
    hr = _clip()
    outs = {k: degrade(hr, 4, kernel=k) for k in ("bicubic", "bilinear", "area")}
    assert not torch.allclose(outs["bicubic"], outs["area"])
    flat = torch.full((2, 3, 64, 64), 0.3)
    for k in outs:
        assert torch.allclose(degrade(flat, 4, kernel=k), torch.full((2, 3, 16, 16), 0.3), atol=1e-6)
    with pytest.raises(ValueError):
        degrade(hr, 4, kernel="lanczos9")


def _write_frames(folder, names, size=(96, 80), seed=0):
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(folder / name)


def _vimeo(tmp_path):
    root, teacher = tmp_path / "vimeo_septuplet", tmp_path / "vimeo_teacher"
    names = [f"im{k}.png" for k in range(1, 8)]
    for i, clip in enumerate(("00001/0001", "00001/0002", "00002/0001")):
        _write_frames(root / "sequences" / clip, names, seed=i)
        _write_frames(teacher / clip, names, seed=i)  # a teacher that returns the frames unchanged
    (root / "sep_trainlist.txt").write_text("00001/0001\n00002/0001\n")
    return root, teacher


def test_vimeo90k_septuplets_follow_the_train_list_and_align_the_teacher(tmp_path):
    root, teacher = _vimeo(tmp_path)
    data = Vimeo90KSeptuplets(root, frames=5, crop=64, teacher_root=teacher)
    assert len(data) == 2
    item = data[1]
    assert item["hr"].shape == item["teacher"].shape == (5, 3, 64, 64)
    assert item["lr"].shape == (5, 3, 16, 16)
    assert torch.equal(item["teacher"], item["hr"])


def test_reds_and_vimeo90k_combine_into_one_training_set(tmp_path):
    root, teacher = _vimeo(tmp_path)
    reds = tmp_path / "REDS"
    _write_frames(reds / "000", [f"{k:08d}.png" for k in range(8)])
    _write_frames(tmp_path / "reds_teacher" / "000", [f"{k:08d}.png" for k in range(8)])
    data = build_dataset({"source": ["reds", "vimeo90k"], "frames": 7, "crop": 64,
                          "reds": {"root": str(reds), "teacher_root": str(tmp_path / "reds_teacher")},
                          "vimeo90k": {"root": str(root), "teacher_root": str(teacher)}}, scale=4)
    assert len(data) == 10 + 2
    shapes = {tuple(data[i]["lr"].shape) for i in (0, len(data) - 1)}
    assert shapes == {(7, 3, 16, 16)}
    synthetic = build_dataset({"source": "synthetic", "clips": 3, "frames": 3, "crop": 64}, scale=4)
    assert len(synthetic) == 3 and synthetic[0]["lr"].shape == (3, 3, 16, 16)
