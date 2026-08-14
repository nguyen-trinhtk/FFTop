// Four-step cache-blocked FFT with OpenMP parallelism.

#include "fft/cpu/parallel-four-step.h"

#include "fft/cpu/detail/four_step.h"

void fft_parallel_four_step(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    output = input;
    FFTCpu::detail::four_step_inplace(output, FFTCpu::detail::parallel_four_step_policy());
}
