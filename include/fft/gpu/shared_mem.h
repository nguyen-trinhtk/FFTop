#pragma once

#include <vector>

#include "fft/core/types.h"

// Shared-memory tiled FFT: fuse early DIT stages in __shared__ for locality.
void fft_gpu_shared_mem(const std::vector<FFTCore::Complex>& input,
                        std::vector<FFTCore::Complex>& output);
