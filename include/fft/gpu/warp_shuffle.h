#pragma once

#include <vector>

#include "fft/core/types.h"

// Warp-shuffle early stages (__shfl_*), then shared-mem tiles, then global.
void fft_gpu_warp_shuffle(const std::vector<FFTCore::Complex>& input,
                          std::vector<FFTCore::Complex>& output);
