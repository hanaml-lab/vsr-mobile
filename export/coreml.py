"""Convert the student to a Core ML package for the iOS demo (iOS 17 or later).

    python -m export.coreml --weights hanaml/vsr-lite --out ios/VSRDemo/VSRLite.mlpackage
"""

from __future__ import annotations

import argparse

import torch

from export.onnx import _Exported
from train.student import STATE_CHANNELS, Student


def convert(model: Student, out: str, height: int = 270, width: int = 480):
    import coremltools as ct

    wrapped = _Exported(model.eval())
    frame = torch.rand(1, 3, height, width)
    state = torch.zeros(1, STATE_CHANNELS, height, width)
    traced = torch.jit.trace(wrapped, (frame, state))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frame", shape=frame.shape), ct.TensorType(name="state", shape=state.shape)],
        outputs=[ct.TensorType(name="upscaled"), ct.TensorType(name="state_out")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.short_description = "vsr-lite: causal x4 video super-resolution with a confidence-gated hidden state"
    mlmodel.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args(argv)
    print(convert(Student.from_pretrained(args.weights), args.out, args.height, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
