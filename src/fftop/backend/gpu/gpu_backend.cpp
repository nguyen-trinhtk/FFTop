#include "fftop/backend/gpu/gpu_backend.h"

#if defined(FFTOP_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace FFTop {

bool GPUBackend::is_available() const {
#if defined(FFTOP_ENABLE_CUDA)
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
    return false;
#endif
}

}  // namespace FFTop
