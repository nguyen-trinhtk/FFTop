#pragma once

#include <vector>

#include "fft/core/types.h"

namespace FFTGpu {
namespace detail {

// How early DIT stages (after bit-reversal) execute:
enum class LocalityMode {
    GlobalOnly,     // before: one global kernel per stage
    WarpThenShared  // after: __shfl_* then __shared__ tiles, then global
};

void fft_iterative_locality(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output,
    LocalityMode mode);

}  // namespace detail
}  // namespace FFTGpu
