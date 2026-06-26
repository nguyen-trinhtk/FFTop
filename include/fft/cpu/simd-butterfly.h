#pragma once

#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace FFTSimd {
namespace detail {

inline void butterfly_scalar(
    std::vector<FFTCore::Complex>& data,
    std::size_t base,
    std::size_t half,
    const std::vector<FFTCore::Complex>& twiddles) {
    for (std::size_t j = 0; j < half; ++j) {
        const std::size_t lo = base + j;
        const std::size_t hi = lo + half;
        const FFTCore::Complex u = data[lo];
        const FFTCore::Complex t = twiddles[j] * data[hi];
        data[lo] = u + t;
        data[hi] = u - t;
    }
}

#if defined(__aarch64__) || defined(__ARM_NEON)

inline float64x2_t cmul_neon(float64x2_t w, float64x2_t z) {
    const double wr = vgetq_lane_f64(w, 0);
    const double wi = vgetq_lane_f64(w, 1);
    const double zr = vgetq_lane_f64(z, 0);
    const double zi = vgetq_lane_f64(z, 1);
    const double result[2] = {wr * zr - wi * zi, wr * zi + wi * zr};
    return vld1q_f64(result);
}

inline void butterfly_group(
    std::vector<FFTCore::Complex>& data,
    std::size_t base,
    std::size_t half,
    const std::vector<FFTCore::Complex>& twiddles) {
    std::size_t j = 0;
    for (; j + 1 < half; j += 2) {
        const std::size_t lo0 = base + j;
        const std::size_t lo1 = base + j + 1;
        const std::size_t hi0 = lo0 + half;
        const std::size_t hi1 = lo1 + half;

        const float64x2_t u0 = vld1q_f64(reinterpret_cast<double*>(&data[lo0]));
        const float64x2_t u1 = vld1q_f64(reinterpret_cast<double*>(&data[lo1]));
        const float64x2_t v0 = vld1q_f64(reinterpret_cast<double*>(&data[hi0]));
        const float64x2_t v1 = vld1q_f64(reinterpret_cast<double*>(&data[hi1]));
        const float64x2_t w0 = vld1q_f64(reinterpret_cast<const double*>(&twiddles[j]));
        const float64x2_t w1 = vld1q_f64(reinterpret_cast<const double*>(&twiddles[j + 1]));

        const float64x2_t t0 = cmul_neon(w0, v0);
        const float64x2_t t1 = cmul_neon(w1, v1);

        vst1q_f64(reinterpret_cast<double*>(&data[lo0]), vaddq_f64(u0, t0));
        vst1q_f64(reinterpret_cast<double*>(&data[hi0]), vsubq_f64(u0, t0));
        vst1q_f64(reinterpret_cast<double*>(&data[lo1]), vaddq_f64(u1, t1));
        vst1q_f64(reinterpret_cast<double*>(&data[hi1]), vsubq_f64(u1, t1));
    }

    for (; j < half; ++j) {
        const std::size_t lo = base + j;
        const std::size_t hi = lo + half;
        const FFTCore::Complex u = data[lo];
        const FFTCore::Complex t = twiddles[j] * data[hi];
        data[lo] = u + t;
        data[hi] = u - t;
    }
}

#else

inline void butterfly_group(
    std::vector<FFTCore::Complex>& data,
    std::size_t base,
    std::size_t half,
    const std::vector<FFTCore::Complex>& twiddles) {
    butterfly_scalar(data, base, half, twiddles);
}

#endif

inline void build_twiddles(std::size_t len, std::vector<FFTCore::Complex>& twiddles) {
    const std::size_t half = len / 2;
    twiddles.resize(half);
    const FFTCore::Real theta = -2.0 * FFTCore::PI / static_cast<FFTCore::Real>(len);
    for (std::size_t j = 0; j < half; ++j) {
        const FFTCore::Real angle = theta * static_cast<FFTCore::Real>(j);
        twiddles[j] = FFTCore::Complex(std::cos(angle), std::sin(angle));
    }
}

inline void fft_stages(std::vector<FFTCore::Complex>& data, bool parallel_groups) {
    const std::size_t N = data.size();
    std::vector<FFTCore::Complex> twiddles;

    for (std::size_t len = 2; len <= N; len <<= 1) {
        const std::size_t half = len / 2;
        build_twiddles(len, twiddles);

        if (parallel_groups) {
#if defined(_OPENMP)
            const int group_count = static_cast<int>(N / len);
#pragma omp parallel for if (group_count >= 4)
            for (int group = 0; group < group_count; ++group) {
                const std::size_t base = static_cast<std::size_t>(group) * len;
                butterfly_group(data, base, half, twiddles);
            }
#else
            for (std::size_t base = 0; base < N; base += len) {
                butterfly_group(data, base, half, twiddles);
            }
#endif
        } else {
            for (std::size_t base = 0; base < N; base += len) {
                butterfly_group(data, base, half, twiddles);
            }
        }
    }
}

}  // namespace detail
}  // namespace FFTSimd
