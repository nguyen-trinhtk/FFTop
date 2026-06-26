#pragma once

#include <vector>

#include "fft/core/types.h"

namespace FFTCpu {
namespace detail {

// In-place Cooley-Tukey radix-2 FFT (scalar, on-the-fly twiddle factors).
void iterative_radix2_inplace(std::vector<FFTCore::Complex>& data);

}  // namespace detail
}  // namespace FFTCpu
