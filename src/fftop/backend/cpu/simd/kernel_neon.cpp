#include "fftop/backend/cpu/simd/arch.h"

#if defined(FFTOP_HAS_NEON_KERNEL)

#include "fftop/backend/cpu/simd/isa_neon.h"
#include "fftop/backend/cpu/simd/kernels.h"

#include <cmath>

namespace FFTop::CPU {
namespace {

namespace ISA = ::FFTop::CPU::Neon;

#include "fftop/backend/cpu/simd/kernel.inl"

}  // namespace

void radix2_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix2(data, offset, stride, dir);
}

void radix4_neon(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix4(data, offset, stride, dir);
}

}  // namespace FFTop::CPU

#endif
