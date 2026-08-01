#pragma once

#include <vector>

#include "fft/core/types.h"

// Warp-shuffle micro-FFT leaves (registers + __shfl_*); six-step for large N.
void fft_gpu_warp_shuffle(const std::vector<FFTCore::Complex>& input,
                          std::vector<FFTCore::Complex>& output);
