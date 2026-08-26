#pragma once

#include "fftop/backend/cpu/simd/arch.h"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

void radix2_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);
void radix4_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);

#if defined(FFTOP_HAS_AVX2_KERNEL)
void radix2_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);
void radix4_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);
#endif

#if defined(FFTOP_HAS_NEON_KERNEL)
void radix2_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);
void radix4_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir);
#endif

}  // namespace FFTop::CPU
