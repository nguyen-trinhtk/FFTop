#pragma once

#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

void radix2_dit_scalar(Complex* data, std::size_t offset, std::size_t stride, double sign);
void radix2_dit_neon  (Complex* data, std::size_t offset, std::size_t stride, double sign);
void radix2_dit_avx2  (Complex* data, std::size_t offset, std::size_t stride, double sign);

inline void radix2_dit(Complex* data, std::size_t offset, std::size_t stride, double sign) {
#if defined(FFTOP_ENABLE_AVX2)
    radix2_dit_avx2(data, offset, stride, sign);
#elif defined(FFTOP_ENABLE_NEON)
    radix2_dit_neon(data, offset, stride, sign);
#else
    radix2_dit_scalar(data, offset, stride, sign);
#endif
}

}  // namespace FFTop::CPU
