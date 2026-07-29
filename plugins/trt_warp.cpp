// VsrWarp: a TensorRT plugin for the backward warp (bilinear sampling at x + flow, border
// padding) that stays in FP16 when the rest of the engine is INT8. Quantising the warp to INT8
// shifted sampled positions by up to half a pixel and showed up as shimmer on edges.
//
// Inputs:  features (N, C, H, W), flow (N, 2, H, W) in pixels.  Output: (N, C, H, W).
// Built as CUDA (see CMakeLists.txt); load with ctypes or trtexec --plugins=libtrt_warp.so.

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

constexpr const char *kName = "VsrWarp";
constexpr const char *kVersion = "1";

template <typename T>
__device__ inline float load(const T *p, int i) { return static_cast<float>(p[i]); }
template <>
__device__ inline float load<__half>(const __half *p, int i) { return __half2float(p[i]); }

template <typename T>
__device__ inline void store(T *p, int i, float v) { p[i] = static_cast<T>(v); }
template <>
__device__ inline void store<__half>(__half *p, int i, float v) { p[i] = __float2half(v); }

template <typename T>
__global__ void warp_kernel(const T *feat, const T *flow, T *out, int n, int c, int h, int w) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * h * w;
    if (idx >= total) return;
    const int x = idx % w, y = (idx / w) % h, b = idx / (w * h);
    const int plane = h * w;
    const float fx = load(flow, (b * 2 + 0) * plane + y * w + x);
    const float fy = load(flow, (b * 2 + 1) * plane + y * w + x);
    const float sx = fminf(fmaxf(x + fx, 0.f), float(w - 1));
    const float sy = fminf(fmaxf(y + fy, 0.f), float(h - 1));
    const int x0 = int(floorf(sx)), y0 = int(floorf(sy));
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float ax = sx - x0, ay = sy - y0;
    for (int ch = 0; ch < c; ++ch) {
        const int base = (b * c + ch) * plane;
        const float v = (1 - ay) * ((1 - ax) * load(feat, base + y0 * w + x0) + ax * load(feat, base + y0 * w + x1)) +
                        ay * ((1 - ax) * load(feat, base + y1 * w + x0) + ax * load(feat, base + y1 * w + x1));
        store(out, base + y * w + x, v);
    }
}

class VsrWarp final : public IPluginV2DynamicExt {
public:
    VsrWarp() = default;

    const char *getPluginType() const noexcept override { return kName; }
    const char *getPluginVersion() const noexcept override { return kVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void *) const noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char *ns) noexcept override { ns_ = ns; }
    const char *getPluginNamespace() const noexcept override { return ns_.c_str(); }

    IPluginV2DynamicExt *clone() const noexcept override {
        auto *p = new VsrWarp();
        p->setPluginNamespace(ns_.c_str());
        return p;
    }

    DataType getOutputDataType(int32_t, const DataType *inputTypes, int32_t) const noexcept override {
        return inputTypes[0];
    }

    DimsExprs getOutputDimensions(int32_t, const DimsExprs *inputs, int32_t, IExprBuilder &) noexcept override {
        return inputs[0];
    }

    // FP16 or FP32, linear layout, both inputs and the output of the same type. INT8 is not
    // offered, which is the point of the plugin.
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *io, int32_t, int32_t) noexcept override {
        const auto &d = io[pos];
        const bool ok = d.format == TensorFormat::kLINEAR && (d.type == DataType::kHALF || d.type == DataType::kFLOAT);
        return ok && (pos == 0 || d.type == io[0].type);
    }

    void configurePlugin(const DynamicPluginTensorDesc *, int32_t, const DynamicPluginTensorDesc *, int32_t) noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc *, int32_t, const PluginTensorDesc *, int32_t) const noexcept override {
        return 0;
    }

    int32_t enqueue(const PluginTensorDesc *in, const PluginTensorDesc *, const void *const *inputs, void *const *outputs,
                    void *, cudaStream_t stream) noexcept override {
        const auto &d = in[0].dims;
        const int n = d.d[0], c = d.d[1], h = d.d[2], w = d.d[3];
        const int total = n * h * w, threads = 256, blocks = (total + threads - 1) / threads;
        if (in[0].type == DataType::kHALF) {
            warp_kernel<__half><<<blocks, threads, 0, stream>>>(static_cast<const __half *>(inputs[0]),
                                                                static_cast<const __half *>(inputs[1]),
                                                                static_cast<__half *>(outputs[0]), n, c, h, w);
        } else {
            warp_kernel<float><<<blocks, threads, 0, stream>>>(static_cast<const float *>(inputs[0]),
                                                               static_cast<const float *>(inputs[1]),
                                                               static_cast<float *>(outputs[0]), n, c, h, w);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

private:
    std::string ns_;
};

class VsrWarpCreator final : public IPluginCreator {
public:
    const char *getPluginName() const noexcept override { return kName; }
    const char *getPluginVersion() const noexcept override { return kVersion; }
    const PluginFieldCollection *getFieldNames() noexcept override { return &fields_; }
    IPluginV2 *createPlugin(const char *, const PluginFieldCollection *) noexcept override { return new VsrWarp(); }
    IPluginV2 *deserializePlugin(const char *, const void *, size_t) noexcept override { return new VsrWarp(); }
    void setPluginNamespace(const char *ns) noexcept override { ns_ = ns; }
    const char *getPluginNamespace() const noexcept override { return ns_.c_str(); }

private:
    PluginFieldCollection fields_{0, nullptr};
    std::string ns_;
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(VsrWarpCreator);
