// Four-step cache-blocked FFT with SIMD butterflies and OpenMP parallelism.

#include "fft/cpu/simd-butterfly.h"
#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ParallelFourStep {
namespace detail {

using FFTCore::Complex;
using FFTCore::Real;

std::size_t log2_floor(std::size_t n) {
    std::size_t bits = 0;
    while (n > 1) {
        n >>= 1;
        ++bits;
    }
    return bits;
}

void fft_openmp_simd_inplace(std::vector<Complex>& data) {
    const std::size_t N = data.size();
    if (N <= 1) {
        return;
    }
    FFTCore::bit_reverse_permute(data);
    FFTSimd::detail::fft_stages(data, true);
}

void fft_simd_inplace(std::vector<Complex>& data) {
    const std::size_t N = data.size();
    if (N <= 1) {
        return;
    }
    FFTCore::bit_reverse_permute(data);
    FFTSimd::detail::fft_stages(data, false);
}

void fft_row_simd(std::vector<Complex>& data, std::size_t offset, std::size_t row_length) {
    if (row_length <= 1) {
        return;
    }

    std::vector<Complex> row(row_length);
    for (std::size_t i = 0; i < row_length; ++i) {
        row[i] = data[offset + i];
    }
    fft_simd_inplace(row);
    for (std::size_t i = 0; i < row_length; ++i) {
        data[offset + i] = row[i];
    }
}

void transpose_blocked(
    const Complex* src,
    Complex* dst,
    std::size_t n_rows,
    std::size_t n_cols,
    std::size_t block = 64) {
#if defined(_OPENMP)
#pragma omp parallel for collapse(2) if (n_rows >= 4 && n_cols >= 4)
#endif
    for (std::size_t row0 = 0; row0 < n_rows; row0 += block) {
        for (std::size_t col0 = 0; col0 < n_cols; col0 += block) {
            const std::size_t row_end = std::min(row0 + block, n_rows);
            const std::size_t col_end = std::min(col0 + block, n_cols);
            for (std::size_t row = row0; row < row_end; ++row) {
                for (std::size_t col = col0; col < col_end; ++col) {
                    dst[col * n_rows + row] = src[row * n_cols + col];
                }
            }
        }
    }
}

void apply_twiddles(Complex* data, std::size_t n1, std::size_t n2) {
    const std::size_t N = n1 * n2;
    const Real inv_n = 1.0 / static_cast<Real>(N);

#if defined(_OPENMP)
#pragma omp parallel for collapse(2) if (n2 >= 4 && n1 >= 4)
#endif
    for (std::size_t row = 0; row < n2; ++row) {
        for (std::size_t col = 0; col < n1; ++col) {
            const std::size_t idx = row * n1 + col;
            const Real angle = -2.0 * FFTCore::PI * static_cast<Real>(row * col) * inv_n;
            data[idx] *= Complex(std::cos(angle), std::sin(angle));
        }
    }
}

std::size_t choose_n1(std::size_t N) {
    const std::size_t log2n = log2_floor(N);
    return std::size_t{1} << (log2n / 2);
}

void fft_parallel_four_step_impl(std::vector<Complex>& data) {
    const std::size_t N = data.size();
    assert(N >= 1 && (N & (N - 1)) == 0);

    if (N <= 1024) {
        fft_openmp_simd_inplace(data);
        return;
    }

    const std::size_t n1 = choose_n1(N);
    const std::size_t n2 = N / n1;

#if defined(_OPENMP)
#pragma omp parallel for if (n1 >= 4)
#endif
    for (int row = 0; row < static_cast<int>(n1); ++row) {
        fft_row_simd(data, static_cast<std::size_t>(row) * n2, n2);
    }

    std::vector<Complex> transposed(N);
    transpose_blocked(data.data(), transposed.data(), n1, n2);

#if defined(_OPENMP)
#pragma omp parallel for if (n2 >= 4)
#endif
    for (int row = 0; row < static_cast<int>(n2); ++row) {
        fft_row_simd(transposed, static_cast<std::size_t>(row) * n1, n1);
    }

    apply_twiddles(transposed.data(), n1, n2);
    data.swap(transposed);
}

}  // namespace

void fft_parallel_four_step_impl(std::vector<FFTCore::Complex>& data) {
    detail::fft_parallel_four_step_impl(data);
}

}  // namespace ParallelFourStep

void fft_parallel_four_step(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output) {
    output = input;
    ParallelFourStep::fft_parallel_four_step_impl(output);
}
