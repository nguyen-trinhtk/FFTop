#pragma once

#include <vector>

#include "fft/core/types.h"

using namespace FFTCore;

void dft(const std::vector<Complex>& input, std::vector<Complex>& output);
