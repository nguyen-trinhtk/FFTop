#pragma once

#include <vector>

#include "fft/core/types.h"

// Bailey six-step on device (transpose → col FFTs → transpose → twiddles →
// row FFTs → transpose), with shared-mem leaves for factors that fit a CTA.
void fft_gpu_four_step(const std::vector<FFTCore::Complex>& input,
                       std::vector<FFTCore::Complex>& output);
