#include "fftop/backend/gpu/gpu_backend.h"

#include <cuda_runtime.h>

namespace FFTop {

bool GPUBackend::is_available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace FFTop
