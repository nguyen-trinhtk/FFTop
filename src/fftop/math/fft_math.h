#pragma once

#include <cstddef>

// When compiled by nvcc the function is available in both host and device code.
#ifdef __CUDACC__
#define FFTOP_HOST_DEVICE __host__ __device__
#else
#define FFTOP_HOST_DEVICE
#endif

namespace FFTop::Math {

// Index with its base-`radix` digits reversed; radix 2 is plain bit reversal.
FFTOP_HOST_DEVICE inline std::size_t digit_reverse(std::size_t index,
                                                    std::size_t n,
                                                    std::size_t radix) {
    std::size_t reversed = 0;
    for (std::size_t rest = n; rest > 1; rest /= radix) {
        reversed = reversed * radix + index % radix;
        index /= radix;
    }
    return reversed;
}

}  // namespace FFTop::Math
