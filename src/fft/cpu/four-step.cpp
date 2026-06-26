// Four-step cache-blocked FFT (serial).

#include "fft/cpu/four-step.h"

#include "fft/cpu/detail/four_step.h"

void fft_four_step(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    output = input;
    FFTCpu::detail::four_step_inplace(output, FFTCpu::detail::serial_four_step_policy());
}
