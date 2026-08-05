# vsr-mobile

Upscales video on the device that plays it, from 270p to 1080p (x4 in each direction). The student network has
about 1.9 million parameters and sees only the current and previous frames, so it can run live; it carries a hidden
state forward through time, warped by optical flow. When the flow is unreliable (a cut, fast motion, an occlusion) a
confidence gate resets that state instead of smearing the old frame into the new one. This is the code behind the
lab's [`vsr-lite`](https://hanaml.com/models/vsr-lite) model.

## Install

```bash
pip install -e ".[train,export]"                        # training and export
cargo build --release -p vsr-runtime --features onnx   # the playback pipeline, with ONNX Runtime
```

TensorRT 10 for Jetson, Core ML for iOS 17 or later, ONNX Runtime elsewhere. The pipeline uses `ffmpeg` and
`ffprobe` for decoding and encoding.

## Quickstart

```bash
# upscale a file on a desktop
python -m export.onnx --weights hanaml/vsr-lite --out vsr.onnx
cargo run --release -p vsr-runtime --features onnx -- --model vsr.onnx --input clip_270p.mp4 --output clip_1080p.mp4

# build an INT8 engine on a Jetson Orin Nano
python -m export.tensorrt --onnx vsr.onnx --int8 --calib data/calib/ --out vsr.plan
```

```python
from train.student import Student
model = Student.from_pretrained("hanaml/vsr-lite").eval()
hidden = None
for frame in frames:                      # (1, 3, H, W) tensors in [0, 1]
    out, hidden = model(frame, hidden)
```

`examples/smoke.sh` runs the whole path on a laptop with no data: a short training run on synthetic clips, ONNX
export, and the Rust pipeline upscaling a generated 270p clip.

## How it works

- **Teacher.** BasicVSR++ produces targets on REDS and Vimeo-90K. It is bidirectional and too slow to run live; its outputs are computed once and read during training (`teacher_root` in `configs/student-x4.yaml`).
- **Student.** A causal recurrent network: features of the new frame, plus the previous hidden state warped by a small flow head (a three-level pyramid in the style of RAFT's update block). The hidden state holds the previous frame and 13 feature channels, so the exported model has exactly two inputs, `frame` and `state`.
- **Gate.** A per-pixel confidence from the flow head decides how much of the warped state to keep; below a threshold the state is reset. The runtime also clears the state at hard cuts it detects from luma histograms.
- **Losses.** Distillation loss against the teacher, plus a temporal-consistency loss on consecutive outputs, which is what keeps flicker down.
- **Runtime.** The Rust pipeline decodes, queues two frames, runs the engine and re-encodes, reading NV12 frames in place from the decoder's buffer. A custom TensorRT plugin (`plugins/trt_warp.cpp`) keeps the warp in FP16 when the rest is INT8.

## Benchmarks

x4 on REDS4, and 270p to 1080p on a Jetson Orin Nano.

| Metric | Baseline | vsr-mobile |
| --- | --- | --- |
| PSNR, REDS4 (x4), vs. bicubic + sharpen | 28.6 dB | 30.9 dB |
| Temporal warping error (lower is better), vs. single-frame Real-ESRGAN-small | 4.1 | 2.3 |
| Sustained frame time after 30 min, INT8 TensorRT | 41 ms | 14 ms |
| Clips where viewers objected to flicker (24 raters, 40 clips) | 38% | 12% |

## Development

```bash
pip install -e ".[train,export,dev]"
pytest
ruff check .
cargo test && cargo clippy --all-targets --features onnx -- -D warnings
```

## Related

- Research: [temporally stable restoration at the edge](https://hanaml.com/research/on-device-video-super-resolution).
- Model: [vsr-lite](https://hanaml.com/models/vsr-lite).

## Licence

Apache-2.0. Teacher weights keep their own licence and are not redistributed.
