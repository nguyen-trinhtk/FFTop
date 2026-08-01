#pragma once

#include <vector>

#include "fft/core/types.h"

namespace FFTGpu {
namespace detail {

// Leaf FFT used when a factor fits in one CTA / warp.
enum class FftLeaf {
    SharedBlock,   // butterflies in __shared__ (N <= 1024)
    WarpShuffle,   // registers + __shfl_* (N <= 32)
};

// Bailey six-step decomposition on device, recursing until the leaf fits.
// Correct for all power-of-2 N; used by four-step and warp-shuffle variants.
void fft_hierarchical(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output,
    FftLeaf leaf);

}  // namespace detail
}  // namespace FFTGpu
