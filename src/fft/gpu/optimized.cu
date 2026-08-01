#include "fft/gpu/optimized.h"

#include "fft/gpu/detail/iterative.h"

void fft_gpu_optimized(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output) {
    // cache / warp / shared-mem / data-flow locality on the same iterative FFT
    FFTGpu::detail::fft_iterative_locality(
        input, output, FFTGpu::detail::LocalityMode::WarpThenShared);
}
