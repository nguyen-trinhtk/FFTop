#pragma once

#include <cstddef>
#include <string>

namespace FFTop {

enum class Simd { Scalar, Sse2, Avx2, Avx512, Neon };

struct SystemConfig {
    std::string cpu;
    std::size_t cpu_threads = 1;
    Simd        simd        = Simd::Scalar;
    Simd        kernel_simd = Simd::Scalar;
    bool        openmp      = false;
    bool        nvidia_gpu  = false;
};

inline const char* to_string(Simd simd) {
    switch (simd) {
    case Simd::Sse2:   return "sse2";
    case Simd::Avx2:   return "avx2";
    case Simd::Avx512: return "avx512";
    case Simd::Neon:   return "neon";
    case Simd::Scalar: return "scalar";
    }
    return "scalar";
}

SystemConfig        detect_system_config();
const SystemConfig& system_config();
std::string         describe_system(const SystemConfig& cfg);

}  // namespace FFTop
