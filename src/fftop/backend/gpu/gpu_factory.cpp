#include "fftop/backend/gpu/gpu_factory.h"

#include "fftop/backend/gpu/gpu_backend.h"
#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"

#include <memory>

namespace FFTop {

namespace {

std::unique_ptr<GPU::IGPURadix> make_gpu_radix(RadixPolicy radix) {
    switch (radix) {
        case RadixPolicy::Radix4:
            return std::make_unique<GPU::GPURadix4>();
        case RadixPolicy::Radix2:
            return std::make_unique<GPU::GPURadix2>();
    }
    return std::make_unique<GPU::GPURadix2>();
}

}  // namespace

std::unique_ptr<IBackend> make_gpu_backend(const FFTPlan& plan) {
    std::unique_ptr<GPUFFTBackend> backend;
    switch (plan.gpu_strategy) {
        case GPUKernelStrategy::StockhamGlobal:
            backend = std::make_unique<GPUFFTBackend>(
                make_gpu_radix(RadixPolicy::Radix2),
                std::make_unique<GPU::StockhamGlobalTraversalStrategy>(),
                "GPU/StockhamGlobal");
            break;
        case GPUKernelStrategy::StockhamShared:
            backend = std::make_unique<GPUFFTBackend>(
                make_gpu_radix(RadixPolicy::Radix2),
                std::make_unique<GPU::StockhamSharedTraversalStrategy>(),
                "GPU/StockhamShared");
            break;
        case GPUKernelStrategy::CooleyTukeyGlobal:
        default:
            backend = std::make_unique<GPUFFTBackend>(
                make_gpu_radix(plan.radix),
                std::make_unique<GPU::IterativeGPUTraversalStrategy>(),
                "GPU/CooleyTukey");
            break;
    }
    if (!backend->is_available()) return {};
    return backend;
}

}  // namespace FFTop
