#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/simd/simd.h"

namespace FFTop::CPU {

void Radix2::butterfly(Buffer& data, std::size_t /*n*/,
                       std::size_t offset, std::size_t stride,
                       Direction dir) const {
    const double sign = (dir == Direction::Forward) ? -1.0 : 1.0;
    radix2_dit(data.data(), offset, stride, sign);
}

void Radix4::butterfly(Buffer& data, std::size_t /*n*/,
                       std::size_t offset, std::size_t stride,
                       Direction dir) const {
    const double sign = (dir == Direction::Forward) ? -1.0 : 1.0;
    // TODO: implement radix-4 DIT butterfly
    (void)data; (void)offset; (void)stride; (void)sign;
}

}  // namespace FFTop::CPU
