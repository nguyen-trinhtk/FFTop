#include "fft/cpu/detail/four_step.h"

#include "fft/cpu/detail/iterative_radix2.h"
#include "fft/cpu/detail/matrix_ops.h"
#include "fft/cpu/detail/simd_radix2.h"
#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <cassert>
#include <vector>

namespace FFTCpu {
namespace detail {
namespace {

void fft_row_inplace(
    std::vector<FFTCore::Complex>& data,
    const std::size_t offset,
    const std::size_t row_length,
    const FourStepPolicy& policy) {
    if (row_length <= 1) {
        return;
    }

    std::vector<FFTCore::Complex> row(row_length);
    for (std::size_t i = 0; i < row_length; ++i) {
        row[i] = data[offset + i];
    }

    if (policy.use_simd) {
        simd_radix2_inplace(row, false);
    } else {
        iterative_radix2_inplace(row);
    }

    for (std::size_t i = 0; i < row_length; ++i) {
        data[offset + i] = row[i];
    }
}

void run_row_ffts(
    std::vector<FFTCore::Complex>& data,
    const std::size_t row_count,
    const std::size_t row_length,
    const FourStepPolicy& policy) {
#if defined(_OPENMP)
    if (policy.parallel_rows) {
#pragma omp parallel for if (row_count >= 4)
        for (int row = 0; row < static_cast<int>(row_count); ++row) {
            fft_row_inplace(
                data,
                static_cast<std::size_t>(row) * row_length,
                row_length,
                policy);
        }
        return;
    }
#endif

    for (std::size_t row = 0; row < row_count; ++row) {
        fft_row_inplace(data, row * row_length, row_length, policy);
    }
}

void small_n_fallback(std::vector<FFTCore::Complex>& data, const FourStepPolicy& policy) {
    if (policy.use_simd) {
        simd_radix2_inplace(data, policy.parallel_rows);
    } else {
        iterative_radix2_inplace(data);
    }
}

}  // namespace

void four_step_inplace(std::vector<FFTCore::Complex>& data, const FourStepPolicy& policy) {
    const std::size_t N = data.size();
    assert(FFTCore::is_power_of_2(N));

    if (N <= policy.small_n_threshold) {
        small_n_fallback(data, policy);
        return;
    }

    const std::size_t n1 = choose_four_step_n1(N);
    const std::size_t n2 = N / n1;

    run_row_ffts(data, n1, n2, policy);

    std::vector<FFTCore::Complex> transposed(N);
    transpose_blocked(
        data.data(),
        transposed.data(),
        n1,
        n2,
        policy.transpose_block_size,
        policy.parallel_transpose);

    run_row_ffts(transposed, n2, n1, policy);

    apply_four_step_twiddles(
        transposed.data(),
        n1,
        n2,
        policy.parallel_twiddles);

    data.swap(transposed);
}

}  // namespace detail
}  // namespace FFTCpu
