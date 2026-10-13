#!/usr/bin/env bash
# The whole path on a laptop, without data or a GPU: a short distillation run on synthetic
# clips, ONNX export, and the Rust pipeline upscaling a generated 270p test clip to 1080p.
# The model this trains is only good enough to show the plumbing works.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$root"

python - "$work" <<'PY'
import sys, yaml, torch
from train.distill import train
from export.onnx import export
cfg = yaml.safe_load(open("configs/student-x4.yaml"))
cfg["data"] = {"source": "synthetic", "clips": 16, "frames": 3, "crop": 32}
cfg["train"].update(steps=20, batch_size=4, workers=0, log_every=10)
model = train(cfg)
export(model, f"{sys.argv[1]}/vsr.onnx", height=270, width=480)
PY

ffmpeg -v error -y -f lavfi -i testsrc2=size=480x270:rate=25 -t 2 -pix_fmt yuv420p "$work/clip_270p.mp4"
cargo run --release -p vsr-runtime --features onnx -- \
  --model "$work/vsr.onnx" --input "$work/clip_270p.mp4" --output "$work/clip_1080p.mp4"
ffprobe -v error -show_entries stream=width,height -of csv=p=0 "$work/clip_1080p.mp4"
