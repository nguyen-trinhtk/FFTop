#pragma once

#include "fftop/types.h"

#include <cstddef>

// Which SIMD kernels this platform builds
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFTOP_AVX2 1
#define FFTOP_AVX512 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define FFTOP_NEON 1
#endif

namespace FFTop::CPU {

// Forward-declare SIMD kernels
struct Scalar;
struct AVX2;
struct AVX512;
struct NEON;

// Generic butterflies over a SIMD kernel, defined in kernels.inl
// instantiated per SIMD extension (in simd/<extension>.cpp)
template <class Kernel>
void radix2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
            const Complex* Ws);

template <class Kernel>
void radix4(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
            const Complex* Ws);

}  // namespace FFTop::CPU
