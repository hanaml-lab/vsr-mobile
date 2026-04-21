"""Export the student to ONNX with two inputs and two outputs:

    frame  (1, 3, h, w)   one low-resolution RGB frame in [0, 1]
    state  (1, 16, h, w)  the hidden state; zeros, or the previous frame's state output
    ->  upscaled (1, 3, 4h, 4w),  state (1, 16, h, w)

    python -m export.onnx --weights hanaml/vsr-lite --out vsr.onnx --height 270 --width 480
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train.student import STATE_CHANNELS, Student


class _Exported(torch.nn.Module):
    """Zero state means "no previous frame": start from the current frame instead."""

    def __init__(self, model: Student):
        super().__init__()
        self.model = model

    def forward(self, frame, state):
        empty = (state.abs().sum() == 0).to(frame.dtype)
        state = empty * self.model.init_state(frame) + (1 - empty) * state
        return self.model(frame, state)


def export(model: Student, out: str | Path, height: int = 270, width: int = 480, opset: int = 17) -> Path:
    model = model.eval()
    frame = torch.rand(1, 3, height, width)
    state = torch.zeros(1, STATE_CHANNELS, height, width)
    torch.onnx.export(_Exported(model), (frame, state), str(out), input_names=["frame", "state"],
                      output_names=["upscaled", "state_out"], opset_version=opset,
                      dynamic_axes={"frame": {2: "h", 3: "w"}, "state": {2: "h", 3: "w"},
                                    "upscaled": {2: "H", 3: "W"}, "state_out": {2: "h", 3: "w"}},
                      dynamo=False)
    return Path(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="a .pt state dict or a Hugging Face repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args(argv)
    print(export(Student.from_pretrained(args.weights), args.out, args.height, args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
