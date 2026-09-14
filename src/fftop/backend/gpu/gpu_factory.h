#pragma once

#include "fftop/backend/ibackend.h"
#include "fftop/plan/plan.h"
#include <memory>

namespace FFTop {
// nullptr when no CUDA dev avail
std::unique_ptr<IBackend> make_gpu_backend(const FFTPlan& plan);
}  // namespace FFTop
