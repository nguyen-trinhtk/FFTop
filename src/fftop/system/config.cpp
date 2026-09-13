#include "fftop/system.h"
#include "fftop/system/detect.h"

#include <sstream>
#include <string>

namespace FFTop {

SystemConfig detect_system_config() {
    SystemConfig cfg;
    cfg.cpu         = detect_cpu_name();
    cfg.cpu_threads = detect_thread_count();
    cfg.simd        = detect_simd();
    cfg.openmp      = detect_openmp();
    cfg.nvidia_gpu  = detect_nvidia_gpu();
    cfg.cuda        = detect_cuda();
    return cfg;
}

const SystemConfig& system_config() {
    static const SystemConfig instance = detect_system_config();
    return instance;
}

std::string describe_system(const SystemConfig& cfg) {
    std::ostringstream out;
    out << "cpu:     " << cfg.cpu << "  (" << cfg.cpu_threads << " threads)\n"
        << "simd:    " << to_string(cfg.simd) << '\n'
        << "openmp:  " << (cfg.openmp ? "yes" : "no") << '\n'
        << "nvidia:  " << (cfg.nvidia_gpu ? "yes" : "no") << '\n'
        << "cuda:    " << (cfg.cuda ? "yes" : "no") << '\n';
    return out.str();
}

}  // namespace FFTop
