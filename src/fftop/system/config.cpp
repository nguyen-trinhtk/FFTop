#include "fftop/system/config.h"

#include <algorithm>
#include <thread>

namespace FFTop {

SystemConfig detect_system_config() {
    SystemConfig config;

    config.cpu_threads      = std::max(1u, std::thread::hardware_concurrency());
    config.enable_parallelism = config.cpu_threads > 1;

    // TODO: auto detect GPU?
    // TODO: also support other GPU backends (metal, vulkan, opengl etc)
#if defined(FFTOP_ENABLE_CUDA)
    int n = 0;
    cudaGetDeviceCount(&n);
    config.has_gpu = n > 0;
#else
    config.has_gpu = false;
#endif

    return config;
}

}  // namespace FFTop
