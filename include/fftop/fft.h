#pragma once

#include "fftop/types.h"
#include <cstddef>

namespace FFTop {
/// @brief Public interface for FFT
/// @param input Host input buffer (size must be power of two)
/// @param options Hardware target and direction
/// @return Output buffer
/// @note Unnormalized in both directions
Buffer fft(const Buffer& input, const FFTOptions& options = {});

/// @brief FFT on device buffer
/// @param d_out Device ptr to output (can be `= d_in`)
/// @param d_in Device ptr to input
/// @param n size
/// @param direction Forward or inverse
void fft_device(Complex* d_out, const Complex* d_in, std::size_t n,
                Direction direction = Direction::Forward);
}  // namespace FFTop
