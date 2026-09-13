#pragma once

#include "fftop/system.h"

#include <string>

namespace FFTop {
    // System config detection
    std::string detect_cpu_name();
    SIMD        detect_simd();
    bool        detect_openmp();
    std::size_t detect_thread_count();
    bool        detect_nvidia_gpu();
    bool        detect_cuda();
}  // namespace FFTop
