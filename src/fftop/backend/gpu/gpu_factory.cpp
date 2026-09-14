#include "fftop/backend/gpu/gpu_factory.h"

#include "fftop/backend/gpu/cooley_tukey.h"
#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"

#include <memory>
#include <utility>

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
    auto backend = std::make_unique<CooleyTukeyGPUBackend>(
        make_gpu_radix(plan.radix),
        std::make_unique<GPU::IterativeGPUTraversalStrategy>());
    if (!backend->is_available()) return {};
    return backend;
}
}  // namespace FFTop
