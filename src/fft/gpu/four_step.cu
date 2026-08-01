#include "fft/gpu/four_step.h"

#include "fft/gpu/detail/hierarchical.h"

void fft_gpu_four_step(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output) {
    FFTGpu::detail::fft_hierarchical(
        input,
        output,
        FFTGpu::detail::FftLeaf::SharedBlock);
}
