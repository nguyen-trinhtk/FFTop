#include "fftop/backend/cpu/simd/kernels.inl"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

// "Scalar-vector" kernel
struct Scalar {
    static constexpr std::size_t width = 1;
    using Pack                         = Complex;

    static Pack load(const Complex& z) { return z; }
    static void store(Complex& z, Pack v) { z = v; }

    static Pack add(Pack a, Pack b) { return a + b; }
    static Pack sub(Pack a, Pack b) { return a - b; }
    static Pack cmul(Pack a, Pack b) { return CPU::cmul(a, b); }
    static Pack mul_j(Pack z) { return {-z.imag(), z.real()}; }
    static Pack mul_minus_j(Pack z) { return {z.imag(), -z.real()}; }
    static Pack conj(Pack z) { return {z.real(), -z.imag()}; }
};

template void radix2<Scalar>(Complex*, std::size_t, Direction, const Complex*);
template void radix4<Scalar>(Complex*, std::size_t, Direction, const Complex*);

}  // namespace FFTop::CPU
