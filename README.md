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
