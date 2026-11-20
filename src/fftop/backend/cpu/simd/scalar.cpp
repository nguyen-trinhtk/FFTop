#include "fftop/backend/cpu/simd/isa_scalar.h"
#include "fftop/backend/cpu/simd/kernels.h"

namespace FFTop::CPU {
namespace {

namespace ISA = ::FFTop::CPU::Scalar;

#include "fftop/backend/cpu/simd/kernel.inl"

}  // namespace

void radix2_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n) {
    radix2(data, offset, stride, dir, W, n);
}

void radix4_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
                   const Complex* W, std::size_t n) {
    radix4(data, offset, stride, dir, W, n);
}

}  // namespace FFTop::CPU
