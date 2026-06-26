#pragma once

#include <cstddef>

#include "fft/core/types.h"

namespace FFTCpu {
namespace detail {

// Pick N1 for four-step decomposition: N = N1 * N2, both powers of 2.
std::size_t choose_four_step_n1(std::size_t N);

// Cache-blocked matrix transpose: dst[col * n_rows + row] = src[row * n_cols + col].
void transpose_blocked(
    const FFTCore::Complex* src,
    FFTCore::Complex* dst,
    std::size_t n_rows,
    std::size_t n_cols,
    std::size_t block_size,
    bool parallel);

// Twiddle multiply for four-step FFT stage.
void apply_four_step_twiddles(
    FFTCore::Complex* data,
    std::size_t n1,
    std::size_t n2,
    bool parallel);

}  // namespace detail
}  // namespace FFTCpu
