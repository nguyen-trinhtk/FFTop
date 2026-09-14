#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/simd/kernels.h"

namespace FFTop::CPU {
namespace {
// SIMD kernel function type
using ButterflyFn = void (*)(Buffer&, std::size_t, std::size_t, Direction, const Complex*, std::size_t);

// Picking SIMD implementation
ButterflyFn pick_radix2(SIMD simd) {
    switch (simd) {
#if defined(FFTOP_HAS_AVX512_KERNEL)
    case SIMD::AVX512:
        return radix2_avx512;
#endif
#if defined(FFTOP_HAS_AVX2_KERNEL)
    case SIMD::AVX2:
        return radix2_avx2;
#endif
#if defined(FFTOP_HAS_NEON_KERNEL)
    case SIMD::NEON:
        return radix2_neon;
#endif
    default:
        return radix2_scalar;
    }
}

ButterflyFn pick_radix4(SIMD simd) {
    switch (simd) {
#if defined(FFTOP_HAS_AVX512_KERNEL)
    case SIMD::AVX512:
        return radix4_avx512;
#endif
#if defined(FFTOP_HAS_AVX2_KERNEL)
    case SIMD::AVX2:
        return radix4_avx2;
#endif
#if defined(FFTOP_HAS_NEON_KERNEL)
    case SIMD::NEON:
        return radix4_neon;
#endif
    default:
        return radix4_scalar;
    }
}

}  // namespace

Radix2::Radix2(SIMD simd) : fn_(pick_radix2(simd)) {}
Radix4::Radix4(SIMD simd) : fn_(pick_radix4(simd)) {}

// Execute
void Radix2::butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                       Direction dir, const Complex* W, std::size_t n) const {
    fn_(data, offset, stride, dir, W, n);
}

void Radix4::butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                       Direction dir, const Complex* W, std::size_t n) const {
    fn_(data, offset, stride, dir, W, n);
}

}  // namespace FFTop::CPU
