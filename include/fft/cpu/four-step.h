#pragma once

#include <vector>

#include "fft/core/types.h"

void fft_four_step(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output);
