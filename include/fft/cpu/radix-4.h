#pragma once

#include <vector>

#include "fft/core/types.h"

void radix_4_fft(const std::vector<FFTCore::Complex>& input,
                 std::vector<FFTCore::Complex>& output);
