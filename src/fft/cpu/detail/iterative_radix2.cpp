#include "fft/cpu/detail/iterative_radix2.h"

#include "fft/core/bitops.h"
#include "fft/core/types.h"

#include <cmath>

namespace FFTCpu {
namespace detail {

void iterative_radix2_inplace(std::vector<FFTCore::Complex>& data) {
    const std::size_t N = data.size();
    if (N <= 1) {
        return;
    }

    FFTCore::bit_reverse_permute(data);

    for (std::size_t len = 2; len <= N; len <<= 1) {
        const FFTCore::Real theta = -2.0 * FFTCore::PI / static_cast<FFTCore::Real>(len);
        const FFTCore::Complex wlen(std::cos(theta), std::sin(theta));

        for (std::size_t i = 0; i < N; i += len) {
            FFTCore::Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                const FFTCore::Complex u = data[i + j];
                const FFTCore::Complex t = w * data[i + j + len / 2];
                data[i + j] = u + t;
                data[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}

}  // namespace detail
}  // namespace FFTCpu
