#include "fft/cpu/detail/iterative_radix2.h"

#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <cmath>

namespace FFTCpu {
namespace detail {
namespace {

void run_butterfly_group(
    std::vector<FFTCore::Complex>& data,
    const std::size_t offset,
    const std::size_t len,
    const FFTCore::Complex& wlen) {
    FFTCore::Complex w(1.0, 0.0);
    for (std::size_t j = 0; j < len / 2; ++j) {
        const FFTCore::Complex u = data[offset + j];
        const FFTCore::Complex t = w * data[offset + j + len / 2];
        data[offset + j] = u + t;
        data[offset + j + len / 2] = u - t;
        w *= wlen;
    }
}

}  // namespace

void iterative_radix2_inplace(std::vector<FFTCore::Complex>& data, const bool parallel_groups) {
    const std::size_t N = data.size();
    if (N <= 1) {
        return;
    }

    FFTCore::bit_reverse_permute(data);

    for (std::size_t len = 2; len <= N; len <<= 1) {
        const FFTCore::Real theta = -2.0 * FFTCore::PI / static_cast<FFTCore::Real>(len);
        const FFTCore::Complex wlen(std::cos(theta), std::sin(theta));
        const std::size_t group_count = N / len;

#if defined(_OPENMP)
        if (parallel_groups) {
#pragma omp parallel for if (group_count >= 4)
            for (int group = 0; group < static_cast<int>(group_count); ++group) {
                run_butterfly_group(data, static_cast<std::size_t>(group) * len, len, wlen);
            }
            continue;
        }
#else
        (void)parallel_groups;
#endif

        for (std::size_t group = 0; group < group_count; ++group) {
            run_butterfly_group(data, group * len, len, wlen);
        }
    }
}

void iterative_radix2_inplace(std::vector<FFTCore::Complex>& data) {
    iterative_radix2_inplace(data, false);
}

}  // namespace detail
}  // namespace FFTCpu
