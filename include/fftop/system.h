#pragma once

#include <cstddef>
#include <string>

namespace FFTop {

    // Supported SIMD instructions
    enum class SIMD { Scalar, AVX2, AVX512, NEON };

    struct SystemConfig {
        // CPU
        std::string cpu;
        std::size_t cpu_threads = 1;
        SIMD        simd        = SIMD::Scalar;
        bool        openmp      = false;

        // GPU
        bool        nvidia_gpu  = false;  // hardware seen (runtime or nvidia-smi)
        bool        cuda        = false;  // CUDA backend compiled in and a device is present
    };


    // Utils: convert simd enum to string
    inline const char* to_string(SIMD simd) {
        switch (simd) {
            case SIMD::AVX2:   return "avx2";
            case SIMD::AVX512: return "avx512";
            case SIMD::NEON:   return "neon";
            case SIMD::Scalar: return "scalar";
        }
        return "unknown";
    }

    SystemConfig        detect_system_config();
    const SystemConfig& system_config();
    std::string         describe_system(const SystemConfig& cfg);
}  // namespace FFTop