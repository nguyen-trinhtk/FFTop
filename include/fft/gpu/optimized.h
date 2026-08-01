#pragma once

#include <vector>

#include "fft/core/types.h"

// After: warp-shuffle + shared-memory tiles + global stages for large spans.
void fft_gpu_optimized(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output);
