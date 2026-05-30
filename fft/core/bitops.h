#pragma once
#include <cstddef>
#include <vector>
#include "fft/core/types.h"

namespace FFTCore {
    // Compute the bit-reversal of `i` for a domain of size N (N must be power of 2)
static std::size_t bit_reverse(std::size_t i, std::size_t N) {
    std::size_t result = 0;
    std::size_t bits = 0;
    std::size_t n = N;
    while (n >>= 1) ++bits;          // bits = log2(N)
    for (std::size_t b = 0; b < bits; ++b) {
        result = (result << 1) | (i & 1);
        i >>= 1;
    }
    return result;
}

// Apply bit-reversal permutation in-place
static void bit_reverse_permute(std::vector<Complex>& a) {
    const std::size_t N = a.size();
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t j = bit_reverse(i, N);
        if (i < j)
            std::swap(a[i], a[j]);
    }
}
}