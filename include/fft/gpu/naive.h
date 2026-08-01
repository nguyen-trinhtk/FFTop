#pragma once

#include <vector>

#include "fft/core/types.h"

// Naive CUDA FFT: global-memory radix-2 stages (one kernel launch per stage).
// Requires a CUDA-capable build (-DFFT_HAS_CUDA, linked with cudart).
void fft_gpu_naive(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output);
