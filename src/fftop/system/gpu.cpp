#include "fftop/system/detect.h"

#include <cstdio>
#include <cstring>

#if defined(FFTOP_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace FFTop {

bool detect_cuda() {
#if defined(FFTOP_ENABLE_CUDA)
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
#else
    return false;
#endif
}

bool detect_nvidia_gpu() {
    if (detect_cuda()) return true;

#if defined(_WIN32)
    FILE* pipe = _popen("nvidia-smi -L 2>NUL", "r");
#else
    FILE* pipe = popen("nvidia-smi -L 2>/dev/null", "r");
#endif
    if (!pipe) return false;

    char line[512];
    bool found = false;
    while (std::fgets(line, sizeof line, pipe)) {
        if (std::strstr(line, "failed") || std::strstr(line, "not found")) {
            found = false;
            break;
        }
        if (std::strstr(line, "GPU")) {
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
