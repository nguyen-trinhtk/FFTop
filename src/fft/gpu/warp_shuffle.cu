#include "fft/gpu/warp_shuffle.h"

#include "fft/gpu/detail/iterative.h"

void fft_gpu_warp_shuffle(const std::vector<FFTCore::Complex>& input,
                          std::vector<FFTCore::Complex>& output) {
    // Stages 2..32 in registers via __shfl_*; 64..1024 in __shared__; rest global.
    FFTGpu::detail::fft_iterative_locality(
        input, output, FFTGpu::detail::LocalityMode::WarpThenShared);
}
