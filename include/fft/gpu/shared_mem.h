#pragma once

#include <vector>

#include "fft/core/types.h"

// Shared-memory block FFT for N that fit a CTA; Stockham DIF for large N.
void fft_gpu_shared_mem(const std::vector<FFTCore::Complex>& input,
                        std::vector<FFTCore::Complex>& output);
