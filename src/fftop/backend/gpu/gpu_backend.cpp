#include "fftop/backend/gpu/gpu_backend.h"

namespace FFTop {

bool GPUBackend::is_available() const {
    return false;  // TODO: probe CUDA device count
}

}  // namespace FFTop
