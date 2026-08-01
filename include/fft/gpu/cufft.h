#pragma once

#include <vector>

#include "fft/core/types.h"

// NVIDIA cuFFT baseline (Z2Z), analogous to FFTW on CPU.
void fft_gpu_cufft(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output);
