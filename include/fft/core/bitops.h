#pragma once

#include <cstddef>
#include <vector>

#include "fft/core/types.h"

namespace FFTCore {

inline bool is_power_of_2(std::size_t n) {
    return n >= 1 && (n & (n - 1)) == 0;
}

inline std::size_t log2_floor(std::size_t n) {
    std::size_t bits = 0;
    while (n > 1) {
        n >>= 1;
        ++bits;
    }
    return bits;
}

// Compute the bit-reversal of `i` for a domain of size N (N must be power of 2).
inline std::size_t bit_reverse(std::size_t i, std::size_t N) {
    std::size_t result = 0;
    std::size_t bits = 0;
    std::size_t n = N;
    while (n >>= 1) {
        ++bits;
    }
    for (std::size_t b = 0; b < bits; ++b) {
        result = (result << 1) | (i & 1);
        i >>= 1;
    }
    return result;
}

// Apply bit-reversal permutation in-place.
inline void bit_reverse_permute(std::vector<Complex>& a) {
    const std::size_t N = a.size();
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t j = bit_reverse(i, N);
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
}

}  // namespace FFTCore
