#include "fftop/backend/gpu/cooley_tukey.h"

#include <cassert>

namespace FFTop {

CooleyTukeyGPUBackend::CooleyTukeyGPUBackend(
    std::unique_ptr<GPU::IGPURadix>             radix,
    std::unique_ptr<GPU::IGPUTraversalStrategy> traversal)
    : radix_(std::move(radix))
    , traversal_(std::move(traversal)) {}

// CUDA implementation lives in gpu_fft.cu and is linked when FFTOP_ENABLE_CUDA
// is set.  Provide a no-op fallback so the library links in CPU-only builds.
#if !defined(FFTOP_ENABLE_CUDA)
void CooleyTukeyGPUBackend::execute(const FFTPlan&, const Buffer& input, Buffer& output) {
    output.resize(input.size());
    assert(false && "CooleyTukeyGPUBackend requires FFTOP_ENABLE_CUDA");
}
#endif

}  // namespace FFTop
