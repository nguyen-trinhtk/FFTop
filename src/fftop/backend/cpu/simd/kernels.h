#pragma once

#include "fftop/types.h"

#include <cstddef>

// Which SIMD kernels

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFTOP_AVX2 1
#define FFTOP_AVX512 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define FFTOP_NEON 1
#endif

namespace FFTop::CPU {

// Single complex multiply. std::complex's operator* adds NaN/inf recovery we don't need.
inline Complex cmul(Complex a, Complex b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

// Forward-declare SIMD kernels

struct Scalar;
struct AVX2;
struct AVX512;
struct NEON;

/// @brief Generic radix-2 butterflies over a SIMD kernel, instantiated per SIMD extension during
/// compile time
/// @tparam Kernel SIMD or Scalar
/// @param data ptr to first element of one butterfly group
/// @param stride stride between elements in a radix block
/// @param dir forward or inverse
/// @param Ws pointer to twiddle table of that stage
template <class Kernel>
void radix2(Complex* data, std::size_t stride, Direction dir, const Complex* Ws);

/// @brief Generic radix-4 butterflies over a SIMD kernel, instantiated per SIMD extension during
/// compile time
/// @tparam Kernel SIMD or Scalar
/// @param data ptr to first element of one butterfly group
/// @param stride stride between elements in a radix block
/// @param dir forward or inverse
/// @param Ws pointer to twiddle table of that stage
template <class Kernel>
void radix4(Complex* data, std::size_t stride, Direction dir, const Complex* Ws);

}  // namespace FFTop::CPU
