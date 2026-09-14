#include "fftop/backend/gpu/cooley_tukey.h"

namespace FFTop {

CooleyTukeyGPUBackend::CooleyTukeyGPUBackend(
    std::unique_ptr<GPU::IGPURadix>             radix,
    std::unique_ptr<GPU::IGPUTraversalStrategy> traversal,
    std::string                                 name)
    : radix_(std::move(radix))
    , traversal_(std::move(traversal))
    , name_(std::move(name)) {}

}  // namespace FFTop
