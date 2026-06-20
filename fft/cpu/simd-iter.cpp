// Iterative in-place radix-2 FFT with precomputed twiddles and SIMD butterflies.

#include "fft/cpu/simd-butterfly.h"
#include "fft/core/types.h"

#include <cassert>
#include <vector>

void fft_simd_iterative(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    const std::size_t N = input.size();
    assert(N >= 1 && (N & (N - 1)) == 0);

    output = input;
    FFTCore::bit_reverse_permute(output);
    FFTSimd::detail::fft_stages(output, false);
}
