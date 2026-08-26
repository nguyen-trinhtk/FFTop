#include "fftop/backend/cpu/simd/arch.h"

#if defined(FFTOP_HAS_AVX2_KERNEL)

#include "fftop/backend/cpu/simd/isa_avx2.h"
#include "fftop/backend/cpu/simd/kernels.h"

#include <cmath>

namespace FFTop::CPU {
namespace {

namespace ISA = ::FFTop::CPU::Avx2;

#include "fftop/backend/cpu/simd/kernel.inl"

}  // namespace

void radix2_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix2(data, offset, stride, dir);
}

void radix4_avx2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    radix4(data, offset, stride, dir);
}

}  // namespace FFTop::CPU

#endif
