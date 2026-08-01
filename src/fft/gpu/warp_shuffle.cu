#include "fft/gpu/warp_shuffle.h"

#include "fft/gpu/detail/hierarchical.h"

void fft_gpu_warp_shuffle(const std::vector<FFTCore::Complex>& input,
                          std::vector<FFTCore::Complex>& output) {
    FFTGpu::detail::fft_hierarchical(
        input,
        output,
        FFTGpu::detail::FftLeaf::WarpShuffle);
}
