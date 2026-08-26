#include "fftop/system.h"
#include "fftop/system/detect.h"
#include "fftop/backend/cpu/simd/arch.h"

#include <sstream>
#include <string>

namespace FFTop {
namespace {

Simd select_kernel(Simd hardware) {
#if defined(FFTOP_HAS_AVX2_KERNEL)
    if (hardware == Simd::Avx2 || hardware == Simd::Avx512) return Simd::Avx2;
#endif
#if defined(FFTOP_HAS_NEON_KERNEL)
    if (hardware == Simd::Neon) return Simd::Neon;
#endif
    return Simd::Scalar;
}

}  // namespace

SystemConfig detect_system_config() {
    SystemConfig cfg;
    cfg.cpu         = detect_cpu_name();
    cfg.cpu_threads = detect_thread_count();
    cfg.simd        = detect_simd();
    cfg.kernel_simd = select_kernel(cfg.simd);
    cfg.openmp      = detect_openmp();
    cfg.nvidia_gpu  = detect_nvidia_gpu();
    return cfg;
}

const SystemConfig& system_config() {
    static const SystemConfig instance = detect_system_config();
    return instance;
}

std::string describe_system(const SystemConfig& cfg) {
    std::ostringstream out;
    out << "cpu:     " << cfg.cpu << "  (" << cfg.cpu_threads << " threads)\n"
        << "simd:    " << to_string(cfg.simd)
        << "  (kernels: " << to_string(cfg.kernel_simd) << ")\n"
        << "openmp:  " << (cfg.openmp ? "yes" : "no") << '\n'
        << "nvidia:  " << (cfg.nvidia_gpu ? "yes" : "no") << '\n';
    return out.str();
}

}  // namespace FFTop
