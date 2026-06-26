#pragma once

#include <cstddef>
#include <vector>

#include "fft/core/types.h"

namespace FFTCpu {
namespace detail {

// Execution policy for the four-step cache-blocked FFT decomposition.
// New backends (e.g. GPU) can add their own policies without duplicating the algorithm skeleton.
struct FourStepPolicy {
    std::size_t small_n_threshold = 1024;
    std::size_t transpose_block_size = 64;
    bool parallel_rows = false;
    bool parallel_transpose = false;
    bool parallel_twiddles = false;
    bool use_simd = false;
};

inline FourStepPolicy serial_four_step_policy() {
    return FourStepPolicy{};
}

inline FourStepPolicy parallel_simd_four_step_policy() {
    FourStepPolicy policy;
    policy.parallel_rows = true;
    policy.parallel_transpose = true;
    policy.parallel_twiddles = true;
    policy.use_simd = true;
    return policy;
}

void four_step_inplace(std::vector<FFTCore::Complex>& data, const FourStepPolicy& policy);

}  // namespace detail
}  // namespace FFTCpu
