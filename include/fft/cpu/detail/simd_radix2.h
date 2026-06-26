#pragma once

#include <vector>

#include "fft/core/types.h"

namespace FFTCpu {
namespace detail {

// In-place radix-2 FFT using precomputed twiddles and SIMD butterflies.
// When parallel_groups is true, butterfly groups are distributed across OpenMP threads.
void simd_radix2_inplace(std::vector<FFTCore::Complex>& data, bool parallel_groups);

}  // namespace detail
}  // namespace FFTCpu
