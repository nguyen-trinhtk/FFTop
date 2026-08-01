#include "fft/gpu/naive.h"

#include "fft/gpu/detail/iterative.h"

void fft_gpu_naive(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output) {
    FFTGpu::detail::fft_iterative_locality(
        input, output, FFTGpu::detail::LocalityMode::GlobalOnly);
}
