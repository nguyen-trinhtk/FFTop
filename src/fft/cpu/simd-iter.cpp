// Iterative in-place radix-2 FFT with precomputed twiddles and SIMD butterflies.

#include "fft/cpu/simd-iter.h"

#include "fft/cpu/detail/simd_radix2.h"
#include "fft/core/bitops.h"

#include <cassert>

void fft_simd_iterative(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    assert(FFTCore::is_power_of_2(input.size()));
    output = input;
    FFTCpu::detail::simd_radix2_inplace(output, false);
}
