// Four-step cache-blocked FFT (self-contained).
// Decomposes N = N1 * N2 into row FFTs, transpose, row FFTs, and twiddle multiply.

#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

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

void fft_iterative_inplace(std::vector<Complex>& data) {
    const std::size_t N = data.size();
    if (N <= 1) {
        return;
    }

    FFTCore::bit_reverse_permute(data);

    for (std::size_t len = 2; len <= N; len <<= 1) {
        const Real theta = -2.0 * FFTCore::PI / static_cast<Real>(len);
        const Complex wlen(std::cos(theta), std::sin(theta));

        for (std::size_t i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex u = data[i + j];
                const Complex t = w * data[i + j + len / 2];
                data[i + j] = u + t;
                data[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}

void fft_row(std::vector<Complex>& data, std::size_t offset, std::size_t row_length) {
    if (row_length <= 1) {
        return;
    }

    std::vector<Complex> row(row_length);
    for (std::size_t i = 0; i < row_length; ++i) {
        row[i] = data[offset + i];
    }
    fft_iterative_inplace(row);
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
    for (std::size_t row0 = 0; row0 < n_rows; row0 += block) {
        const std::size_t row_end = std::min(row0 + block, n_rows);
        for (std::size_t col0 = 0; col0 < n_cols; col0 += block) {
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

void fft_four_step_impl(std::vector<Complex>& data) {
    const std::size_t N = data.size();
    assert(N >= 1 && (N & (N - 1)) == 0);

    if (N <= 1024) {
        fft_iterative_inplace(data);
        return;
    }

    const std::size_t n1 = choose_n1(N);
    const std::size_t n2 = N / n1;

    for (std::size_t row = 0; row < n1; ++row) {
        fft_row(data, row * n2, n2);
    }

    std::vector<Complex> transposed(N);
    transpose_blocked(data.data(), transposed.data(), n1, n2);

    for (std::size_t row = 0; row < n2; ++row) {
        fft_row(transposed, row * n1, n1);
    }

    apply_twiddles(transposed.data(), n1, n2);
    data.swap(transposed);
}

}  // namespace

void fft_four_step(const std::vector<FFTCore::Complex>& input, std::vector<FFTCore::Complex>& output) {
    output = input;
    fft_four_step_impl(output);
}
