#pragma once

#include <vector>

#include "fft/core/types.h"

void radix_2_fft(const std::vector<FFTCore::Complex>& input,
                 std::vector<FFTCore::Complex>& output);
