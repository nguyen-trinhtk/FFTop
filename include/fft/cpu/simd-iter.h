#pragma once

#include <vector>

#include "fft/core/types.h"

void fft_simd_iterative(const std::vector<FFTCore::Complex>& input,
                        std::vector<FFTCore::Complex>& output);
