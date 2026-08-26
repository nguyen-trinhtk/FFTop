#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/simd/kernels.h"

namespace FFTop::CPU {
namespace {

using ButterflyFn = void (*)(Buffer&, std::size_t, std::size_t, Direction);

ButterflyFn pick_radix2(Simd simd) {
    switch (simd) {
#if defined(FFTOP_HAS_AVX2_KERNEL)
    case Simd::Avx2:
    case Simd::Avx512:
        return radix2_avx2;
#endif
#if defined(FFTOP_HAS_NEON_KERNEL)
    case Simd::Neon:
        return radix2_neon;
#endif
    default:
        return radix2_scalar;
    }
}

ButterflyFn pick_radix4(Simd simd) {
    switch (simd) {
#if defined(FFTOP_HAS_AVX2_KERNEL)
    case Simd::Avx2:
    case Simd::Avx512:
        return radix4_avx2;
#endif
#if defined(FFTOP_HAS_NEON_KERNEL)
    case Simd::Neon:
        return radix4_neon;
#endif
    default:
        return radix4_scalar;
    }
}

}  // namespace

Radix2::Radix2(Simd simd) : fn_(pick_radix2(simd)) {}
Radix4::Radix4(Simd simd) : fn_(pick_radix4(simd)) {}

void Radix2::butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                       Direction dir) const {
    fn_(data, offset, stride, dir);
}

void Radix4::butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                       Direction dir) const {
    fn_(data, offset, stride, dir);
}

}  // namespace FFTop::CPU
