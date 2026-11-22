#include "fftop/system.h"
#include "fftop/system/detect.h"

#include <sstream>
#include <string>

namespace FFTop {
namespace {
std::string bytes_str(std::size_t b) {
    // Bytes to string
    if (b == 0) return "?";
    if (b % (1u << 20) == 0) return std::to_string(b >> 20) + "M";
    if (b % (1u << 10) == 0) return std::to_string(b >> 10) + "K";
    return std::to_string(b) + "B";
}
}  // namespace

SystemConfig detect_system_config() {
    SystemConfig cfg;
    cfg.cpu                 = detect_cpu_name();
    cfg.cpu_threads         = detect_thread_count();
    cfg.cpu_cores           = detect_core_count();
    const CacheSizes caches = detect_cache_sizes();
    cfg.l1d_bytes           = caches.l1d;
    cfg.l2_bytes            = caches.l2;
    cfg.l3_bytes            = caches.l3;
    cfg.simd                = detect_simd();
    cfg.openmp              = detect_openmp();
    cfg.nvidia_gpu          = detect_nvidia_gpu();
    cfg.cuda                = detect_cuda();
    if (cfg.cuda) cfg.gpu = detect_gpu_info();
    return cfg;
}

const SystemConfig& system_config() {
    static const SystemConfig instance = detect_system_config();
    return instance;
}

std::string describe_system(const SystemConfig& cfg) {
    std::ostringstream out;
    out << "cpu:     " << cfg.cpu << "  (" << cfg.cpu_threads << " threads, "
        << (cfg.cpu_cores ? std::to_string(cfg.cpu_cores) : std::string("?")) << " cores)\n"
        << "cache:   L1d " << bytes_str(cfg.l1d_bytes) << "  L2 " << bytes_str(cfg.l2_bytes)
        << "  L3 " << bytes_str(cfg.l3_bytes) << '\n'
        << "simd:    " << to_string(cfg.simd) << '\n'
        << "openmp:  " << (cfg.openmp ? "yes" : "no") << '\n'
        << "nvidia:  " << (cfg.nvidia_gpu ? "yes" : "no") << '\n'
        << "cuda:    " << (cfg.cuda ? "yes" : "no") << '\n';
    if (cfg.cuda) {
        const GPUInfo& g = cfg.gpu;
        out << "gpu:     " << g.name << "  (cc " << g.cc_major << '.' << g.cc_minor << ", "
            << g.sm_count << " SMs, " << bytes_str(g.smem_per_block) << " smem/block, "
            << g.max_threads_per_block << " threads/block)\n";
    }
    return out.str();
}
}  // namespace FFTop
