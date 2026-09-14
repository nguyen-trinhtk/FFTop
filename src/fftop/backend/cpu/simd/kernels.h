#pragma once

#include "fftop/backend/cpu/simd/arch.h"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

void radix2_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n);
void radix4_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n);

#if defined(FFTOP_HAS_AVX2_KERNEL)
void radix2_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                 const Complex* W, std::size_t n);
void radix4_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                 const Complex* W, std::size_t n);
#endif

#if defined(FFTOP_HAS_AVX512_KERNEL)
void radix2_avx512(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n);
void radix4_avx512(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n);
#endif

#if defined(FFTOP_HAS_NEON_KERNEL)
void radix2_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                 const Complex* W, std::size_t n);
void radix4_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                 const Complex* W, std::size_t n);
#endif

}  // namespace FFTop::CPU
