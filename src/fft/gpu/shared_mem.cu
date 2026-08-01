#include "fft/gpu/shared_mem.h"

#include "fft/gpu/detail/iterative.h"

void fft_gpu_shared_mem(const std::vector<FFTCore::Complex>& input,
                        std::vector<FFTCore::Complex>& output) {
    // Early DIT stages (len <= 1024) fused in __shared__: one load/store per tile
    // instead of one global round-trip per stage.
    FFTGpu::detail::fft_iterative_locality(
        input, output, FFTGpu::detail::LocalityMode::SharedTile);
}
