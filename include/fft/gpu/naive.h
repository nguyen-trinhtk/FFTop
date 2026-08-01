#pragma once

#include <vector>

#include "fft/core/types.h"

// Naive CUDA FFT (before): bit-reverse + one global radix-2 kernel per stage.
void fft_gpu_naive(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output);
