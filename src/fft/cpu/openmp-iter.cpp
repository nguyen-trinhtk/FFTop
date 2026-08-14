// Iterative in-place radix-2 FFT with OpenMP across butterfly groups.

#include "fft/cpu/openmp-iter.h"

#include "fft/cpu/detail/iterative_radix2.h"
#include "fft/core/bitops.h"

#include <cassert>

void fft_openmp_iterative(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    assert(FFTCore::is_power_of_2(input.size()));
    output = input;
    FFTCpu::detail::iterative_radix2_inplace(output, true);
}
