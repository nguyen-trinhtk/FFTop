#pragma once

#include <cstddef>

// CUDA: available both host & device
#ifdef __CUDACC__
#define FFTOP_HOST_DEVICE __host__ __device__
#else
#define FFTOP_HOST_DEVICE
#endif

namespace FFTop::Math {
namespace {
// check if n is a power of 2
FFTOP_HOST_DEVICE 
inline bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

// n must be non-zero
// return number of trailing 0 bits
FFTOP_HOST_DEVICE 
inline unsigned trailing_zeros(std::size_t n) {
    unsigned c = 0;
    while ((n & 1) == 0) {
        n >>= 1;
        ++c;
    }
    return c;
}
}  // namespace 

FFTOP_HOST_DEVICE 
inline bool is_power_of(std::size_t n, std::size_t base) {
    if (n == 0 || base < 2) return false;
    // fast path for power of 2
    if (is_power_of_two(base)) {
        if (!is_power_of_two(n)) return false;
        const unsigned base_exp = trailing_zeros(base);
        const unsigned n_exp    = trailing_zeros(n);
        return n_exp % base_exp == 0;
    }
    // general case
    while (n % base == 0)
        n /= base;
    return n == 1;
}


// reverse the digits of index in base radix
FFTOP_HOST_DEVICE 
inline std::size_t digit_reverse(std::size_t index, std::size_t n, std::size_t radix) {
    // fast path for power of 2
    if (is_power_of_two(radix)) {
        const std::size_t digit_mask = radix - 1;
        const unsigned    digit_bits = trailing_zeros(radix);
        std::size_t       reversed   = 0;
        for (std::size_t rest = n; rest > 1; rest >>= digit_bits) {
            reversed = (reversed << digit_bits) | (index & digit_mask);
            index >>= digit_bits;
        }
        return reversed;
    }

    // general case
    std::size_t reversed = 0;
    for (std::size_t rest = n; rest > 1; rest /= radix) {
        reversed = reversed * radix + index % radix;
        index /= radix;
    }
    return reversed;
}
}  // namespace FFTop::Math
