#include "fftop/backend/cpu/simd/isa_scalar.h"
#include "fftop/backend/cpu/simd/kernels.h"

#include <cmath>

namespace FFTop::CPU {
namespace {

namespace ISA = ::FFTop::CPU::Scalar;

#include "fftop/backend/cpu/simd/kernel.inl"

}  // namespace

void radix2_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix2(data, offset, stride, dir);
}

void radix4_scalar(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix4(data, offset, stride, dir);
}

}  // namespace FFTop::CPU
