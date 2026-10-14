import pytest
import torch

from train.qat import QuantConv2d, fake_quantize, freeze, prepare_qat
from train.student import Student


def _convs(model, kind=torch.nn.Conv2d):
    return {name for name, m in model.named_modules() if type(m) is kind}


def test_fake_quantize_rounds_to_the_int8_grid_and_clips():
    x = torch.tensor([0.26, -1.0, 5.0, -20.0], requires_grad=True)
    q = fake_quantize(x, torch.tensor(0.1), bits=8)
    assert q.tolist() == pytest.approx([0.3, -1.0, 5.0, -12.8])  # -200 steps clips to -128
    q.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0, 0.0]  # straight through inside the range, zero where clipped


def test_prepare_qat_quantises_every_conv_but_the_flow_head():
    model = Student(ch=16, blocks=2)
    plain = _convs(model)
    prepare_qat(model, bits=8)
    flow = {n for n in plain if n.startswith("flow.")}
    assert flow and _convs(model) == flow  # the flow head stays FP16 on device, so it is not quantised
    assert _convs(model, QuantConv2d) == plain - flow


def test_eight_bit_qat_stays_close_to_the_float_model():
    torch.manual_seed(0)
    model = Student(ch=16, blocks=2).eval()
    frame = torch.rand(1, 3, 12, 20)
    with torch.no_grad():
        ref, _ = model(frame)
        prepare_qat(model, bits=8)
        out, _ = model(frame)
    assert (out - ref).abs().mean() < 0.01


def _levels(weight):
    return max(len(torch.unique(weight[c])) for c in range(weight.shape[0]))


def test_freeze_leaves_plain_convs_whose_weights_sit_on_the_grid():
    torch.manual_seed(0)
    model = Student(ch=16, blocks=2)
    assert _levels(model.body[0].c1.weight) == 16 * 9  # float weights: every one distinct
    prepare_qat(model, bits=4)
    with torch.no_grad():
        model(torch.rand(1, 3, 12, 20))  # activation steps are set on the first batch
    frozen = freeze(model)
    assert not _convs(frozen, QuantConv2d)
    assert _levels(frozen.body[0].c1.weight) <= 16  # 4-bit weights: at most 16 levels per channel


def test_training_with_a_qat_phase_returns_a_frozen_student():
    torch.manual_seed(0)
    cfg = {"model": {"channels": 16, "blocks": 2, "scale": 4, "reset_threshold": 0.35},
           "data": {"source": "synthetic", "clips": 4, "frames": 3, "crop": 32},
           "train": {"batch_size": 2, "lr": 1e-3, "steps": 2, "log_every": 1000},
           "qat": {"bits": 4, "steps": 2, "lr": 1e-4},
           "loss": {"teacher": 1.0, "gt": 0.25, "temporal": 0.5}}
    from train.distill import train

    model = train(cfg)
    assert isinstance(model, Student) and not _convs(model, QuantConv2d)
    assert _levels(model.head.weight) <= 16
