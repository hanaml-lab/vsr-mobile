"""Build a TensorRT engine from the ONNX export, INT8 with calibration, keeping the warp
(GridSample) and the flow head in FP16 so the motion estimate does not lose precision.

    python -m export.tensorrt --onnx vsr.onnx --int8 --calib data/calib/ --out vsr.plan

Run on the target device (TensorRT 10, e.g. a Jetson Orin Nano): engines are not portable.
``plugins/trt_warp.cpp`` provides an FP16 warp for TensorRT versions whose GridSample has no
FP16 kernel; load it with --plugin.
"""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

FP16_LAYERS = ("GridSample", "flow")


def calibration_batches(folder: Path, height: int, width: int, limit: int = 200):
    """Low-resolution frames from PNG files, in a fixed order, with a zero state."""
    from PIL import Image

    for p in sorted(folder.glob("*.png"))[:limit]:
        img = Image.open(p).convert("RGB").resize((width, height), Image.BICUBIC)
        frame = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        yield {"frame": np.ascontiguousarray(frame), "state": np.zeros((1, 16, height, width), np.float32)}


def build(onnx: Path, out: Path, int8: bool, calib: Path | None, height: int, width: int, plugin: Path | None) -> Path:
    import tensorrt as trt

    if plugin:
        ctypes.CDLL(str(plugin))
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for name, ch in (("frame", 3), ("state", 16)):
        shape = (1, ch, height, width)
        profile.set_shape(name, shape, shape, shape)
    config.add_optimization_profile(profile)
    if int8:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.batches = iter(calibration_batches(calib, height, width))
                self.buffers = {}

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                try:
                    batch = next(self.batches)
                except StopIteration:
                    return None
                ptrs = []
                for n in names:
                    if n not in self.buffers:
                        self.buffers[n] = cuda.mem_alloc(batch[n].nbytes)
                    cuda.memcpy_htod(self.buffers[n], batch[n])
                    ptrs.append(int(self.buffers[n]))
                return ptrs

            def read_calibration_cache(self):
                return None

            def write_calibration_cache(self, cache):
                out.with_suffix(".calib").write_bytes(cache)

        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.int8_calibrator = Calibrator()
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if any(k in layer.name for k in FP16_LAYERS):
                layer.precision = trt.float16
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.float16)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT build failed")
    out.write_bytes(engine)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", type=Path)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--plugin", type=Path, help="libtrt_warp.so built from plugins/")
    args = ap.parse_args(argv)
    if args.int8 and not args.calib:
        ap.error("--int8 needs --calib")
    print(build(args.onnx, args.out, args.int8, args.calib, args.height, args.width, args.plugin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
