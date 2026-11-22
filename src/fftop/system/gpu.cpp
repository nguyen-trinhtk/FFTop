#include "fftop/system/detect.h"

#include <cstdio>
#include <cstring>

#if defined(FFTOP_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

// Note: currently only support NVIDIA GPUs with CUDA
namespace FFTop {
bool detect_cuda() {
#if defined(FFTOP_ENABLE_CUDA)
    // Check if CUDA device is available
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

GPUInfo detect_gpu_info() {
    GPUInfo info;
#if defined(FFTOP_ENABLE_CUDA)
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return info;
    info.name = prop.name;
    // Compute capability
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    // Block-related info (sm, smem, max threads)
    info.sm_count              = prop.multiProcessorCount;
    info.smem_per_block        = prop.sharedMemPerBlock;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
#endif
    return info;
}

bool detect_nvidia_gpu() {
    if (detect_cuda()) return true;  // cuda available
#if defined(_WIN32)
    // Windows
    FILE* pipe = _popen("nvidia-smi -L 2>NUL", "r");
#else
    // Unix-like
    FILE* pipe = popen("nvidia-smi -L 2>/dev/null", "r");
#endif
    if (!pipe) return false;

    // Scan output for GPU presence
    char line[512];
    bool found = false;
    while (std::fgets(line, sizeof line, pipe)) {
        if (std::strstr(line, "failed") || std::strstr(line, "not found")) {
            // failed to find GPU
            found = false;
            break;
        }
        if (std::strstr(line, "GPU")) {
            // found GPU
            found = true;
            break;
        }
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return found;
}
}  // namespace FFTop
