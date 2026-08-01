#pragma once

#include <vector>

#include "fft/core/types.h"

// Blocked shared-tile locality path (same engine as shared-mem until Bailey returns).
void fft_gpu_four_step(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output);
