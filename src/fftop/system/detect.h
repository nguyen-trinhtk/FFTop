#pragma once

#include "fftop/system.h"

#include <string>

namespace FFTop {
    struct CacheSizes {
        std::size_t l1d = 0;
        std::size_t l2  = 0;
        std::size_t l3  = 0;
    };
    // System config detection
    std::string detect_cpu_name();
    SIMD        detect_simd();
    bool        detect_openmp();
    std::size_t detect_thread_count();
    std::size_t detect_core_count();   // 0 if unknown
    CacheSizes  detect_cache_sizes();  // 0 for unknown levels
    bool        detect_nvidia_gpu();
    bool        detect_cuda();
    GPUInfo     detect_gpu_info();     // all 0s without CUDA
}  // namespace FFTop
