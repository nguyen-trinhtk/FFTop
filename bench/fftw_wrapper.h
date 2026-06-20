#pragma once

#include <fft/core/types.h>

void fftw_fft_cold(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output);

void fftw_fft_steady(const std::vector<FFTCore::Complex>& input,
                     std::vector<FFTCore::Complex>& output);
