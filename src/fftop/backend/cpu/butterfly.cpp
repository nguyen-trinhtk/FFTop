#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/simd/kernels.h"

namespace FFTop::CPU {
namespace {
// SIMD kernel function type
using ButterflyFn = void (*)(Complex*, std::size_t, Direction, const Complex*);

// Picking SIMD implementation
ButterflyFn pick_radix2(SIMD simd) {
    switch (simd) {
#if defined(FFTOP_AVX512)
        case SIMD::AVX512:
            return radix2<AVX512>;
#endif
#if defined(FFTOP_AVX2)
        case SIMD::AVX2:
            return radix2<AVX2>;
#endif
#if defined(FFTOP_NEON)
        case SIMD::NEON:
            return radix2<NEON>;
#endif
        default:
            return radix2<Scalar>;
    }
}

ButterflyFn pick_radix4(SIMD simd) {
    switch (simd) {
#if defined(FFTOP_AVX512)
        case SIMD::AVX512:
            return radix4<AVX512>;
#endif
#if defined(FFTOP_AVX2)
        case SIMD::AVX2:
            return radix4<AVX2>;
#endif
#if defined(FFTOP_NEON)
        case SIMD::NEON:
            return radix4<NEON>;
#endif
        default:
            return radix4<Scalar>;
    }
}

}  // namespace

Radix2::Radix2(SIMD simd)
    : fn_(pick_radix2(simd)) {}
Radix4::Radix4(SIMD simd)
    : fn_(pick_radix4(simd)) {}

// Execute
void Radix2::butterfly(Complex* data, std::size_t stride, Direction dir, const Complex* Ws) const {
    fn_(data, stride, dir, Ws);
}

void Radix4::butterfly(Complex* data, std::size_t stride, Direction dir, const Complex* Ws) const {
    fn_(data, stride, dir, Ws);
}
}  // namespace FFTop::CPU
