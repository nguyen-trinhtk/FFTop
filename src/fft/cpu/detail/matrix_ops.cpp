#include "fft/cpu/detail/matrix_ops.h"

#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <algorithm>
#include <cmath>

namespace FFTCpu {
namespace detail {

std::size_t choose_four_step_n1(std::size_t N) {
    const std::size_t log2n = FFTCore::log2_floor(N);
    return std::size_t{1} << (log2n / 2);
}

void transpose_blocked(
    const FFTCore::Complex* src,
    FFTCore::Complex* dst,
    const std::size_t n_rows,
    const std::size_t n_cols,
    const std::size_t block_size,
    const bool parallel) {
#if defined(_OPENMP)
    if (parallel) {
#pragma omp parallel for collapse(2) if (n_rows >= 4 && n_cols >= 4)
        for (std::size_t row0 = 0; row0 < n_rows; row0 += block_size) {
            for (std::size_t col0 = 0; col0 < n_cols; col0 += block_size) {
                const std::size_t row_end = std::min(row0 + block_size, n_rows);
                const std::size_t col_end = std::min(col0 + block_size, n_cols);
                for (std::size_t row = row0; row < row_end; ++row) {
                    for (std::size_t col = col0; col < col_end; ++col) {
                        dst[col * n_rows + row] = src[row * n_cols + col];
                    }
                }
            }
        }
        return;
    }
#else
    (void)parallel;
#endif

    for (std::size_t row0 = 0; row0 < n_rows; row0 += block_size) {
        const std::size_t row_end = std::min(row0 + block_size, n_rows);
        for (std::size_t col0 = 0; col0 < n_cols; col0 += block_size) {
            const std::size_t col_end = std::min(col0 + block_size, n_cols);
            for (std::size_t row = row0; row < row_end; ++row) {
                for (std::size_t col = col0; col < col_end; ++col) {
                    dst[col * n_rows + row] = src[row * n_cols + col];
                }
            }
        }
    }
}

void apply_four_step_twiddles(
    FFTCore::Complex* data,
    const std::size_t n1,
    const std::size_t n2,
    const bool parallel) {
    const std::size_t N = n1 * n2;
    const FFTCore::Real inv_n = 1.0 / static_cast<FFTCore::Real>(N);

#if defined(_OPENMP)
    if (parallel) {
#pragma omp parallel for collapse(2) if (n2 >= 4 && n1 >= 4)
        for (std::size_t row = 0; row < n2; ++row) {
            for (std::size_t col = 0; col < n1; ++col) {
                const std::size_t idx = row * n1 + col;
                const FFTCore::Real angle =
                    -2.0 * FFTCore::PI * static_cast<FFTCore::Real>(row * col) * inv_n;
                data[idx] *= FFTCore::Complex(std::cos(angle), std::sin(angle));
            }
        }
        return;
    }
#else
    (void)parallel;
#endif

    for (std::size_t row = 0; row < n2; ++row) {
        for (std::size_t col = 0; col < n1; ++col) {
            const std::size_t idx = row * n1 + col;
            const FFTCore::Real angle =
                -2.0 * FFTCore::PI * static_cast<FFTCore::Real>(row * col) * inv_n;
            data[idx] *= FFTCore::Complex(std::cos(angle), std::sin(angle));
        }
    }
}

}  // namespace detail
}  // namespace FFTCpu
