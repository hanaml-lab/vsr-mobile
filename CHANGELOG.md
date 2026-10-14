# Changelog

## 0.6.0 (2026-10-14)

- training data: Vimeo-90K septuplets beside REDS; H.264 and AV1 compression over whole clips, and bicubic, bilinear
  or area downscaling, in the degradation
- DRIFT's training terms (`train/drift.py`, `configs/student-x4-drift.yaml`): contraction penalty with a
  power-iteration norm, consistency matching, a gradient-domain texture term and the drift bound
- quantisation-aware fine-tuning with learned step sizes (`train/qat.py`), the flow head left in floating point
- runtime: `--frame-budget-ms` and `--fallback-frames`, a frame-time budget with a bicubic fallback
- fix: frames too small for the flow pyramid raise an error instead of warping with a NaN grid
- `data.crop` is high-res pixels for every source, synthetic clips included
- README with results, papers and citation; `CITATION.cff`; `CONTRIBUTING.md`; CI with ffmpeg codecs, read-only
  permissions and one run per branch

## 0.5.0 (2026-10-06)

- TensorRT INT8 export with an FP16 warp plugin
- Core ML demo
- smoke example

## 0.3.0 (2026-04-29)

- playback pipeline with ONNX Runtime and scene-cut resets

## 0.1.0 (2026-04-22)

- student network, distillation losses and ONNX export
