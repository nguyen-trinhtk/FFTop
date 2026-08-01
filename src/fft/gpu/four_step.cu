#include "fft/gpu/four_step.h"

#include "fft/gpu/detail/iterative.h"

void fft_gpu_four_step(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output) {
    // Same blocked shared-tile locality as shared-mem for now (no Bailey recursion).
    // Keeps the registry slot for a real four-step later.
    FFTGpu::detail::fft_iterative_locality(
        input, output, FFTGpu::detail::LocalityMode::SharedTile);
}
