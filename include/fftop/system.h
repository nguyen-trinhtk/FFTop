#pragma once

#include <cstddef>
#include <string>

namespace FFTop {

// Supported SIMD instructions
enum class SIMD { Scalar, AVX2, AVX512, NEON };

// Facts about the first CUDA device; all zero when CUDA is unusable.
struct GPUInfo {
    std::string name;
    int cc_major = 0;  // compute capability
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t smem_per_block = 0;  // default limit (bytes)
    std::size_t smem_per_block_optin = 0;  // opt-in limit for dynamic smem (bytes)
    std::size_t smem_per_sm = 0;  // bytes
    std::size_t reserved_smem_per_block = 0;  // reserved by runtime
    int max_threads_per_block = 0;
    int max_threads_per_sm = 0;
};

// Hardware facts, detected once per process. 0 means "unknown";
// the planners fall back to defaults (see plan/planner.h).
struct SystemConfig {
    // CPU
    std::string cpu;
    std::size_t cpu_threads = 1;  // logical
    std::size_t cpu_cores = 0;    // physical
    std::size_t l1d_bytes = 0;    // per core
    std::size_t l2_bytes = 0;
    std::size_t l3_bytes = 0;
    SIMD simd = SIMD::Scalar;
    bool openmp = false;

    // GPU
    bool nvidia_gpu = false;  // gpu discovered
    bool cuda = false;        // cuda compiled well
    GPUInfo gpu;
};

// Utils: convert simd enum to string
inline const char* to_string(SIMD simd) {
    switch (simd) {
        case SIMD::AVX2:
            return "avx2";
        case SIMD::AVX512:
            return "avx512";
        case SIMD::NEON:
            return "neon";
        case SIMD::Scalar:
            return "scalar";
    }
    return "unknown";
}

SystemConfig detect_system_config();
const SystemConfig& system_config();
std::string describe_system(const SystemConfig& cfg);
}  // namespace FFTop
