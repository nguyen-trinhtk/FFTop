#pragma once

#include <vector>

#include "fft/core/types.h"

namespace FFTGpu {
namespace detail {

// How early DIT stages (after bit-reversal) execute:
enum class LocalityMode {
    GlobalOnly,     // one global kernel per stage (naive)
    SharedTile,     // fuse stages 2..tile in __shared__ (tile <= 1024)
    WarpThenShared  // fuse 2..32 via __shfl_*, then 64..tile in __shared__
};

void fft_iterative_locality(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output,
    LocalityMode mode);

}  // namespace detail
}  // namespace FFTGpu
