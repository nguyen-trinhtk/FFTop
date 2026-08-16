#pragma once

#include <cstddef>

namespace FFTop {

struct SystemConfig {
    std::size_t cpu_threads       = 1;
    bool has_gpu           = false;
    bool enable_parallelism = true;
    // TODO: SIMD type, cache size, memory bandwidth, gflops, etc.
};

SystemConfig detect_system_config();

}  // namespace FFTop
