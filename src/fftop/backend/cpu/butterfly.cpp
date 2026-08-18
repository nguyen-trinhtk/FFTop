#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/simd/isa.h"

#include <cmath>

namespace FFTop::CPU {

// Radix-2 DIT butterfly
void Radix2::butterfly(Buffer& data, std::size_t /*n*/,
                       std::size_t offset, std::size_t stride,
                       Direction dir) const {
    const double sign = (dir == Direction::Forward) ? -1.0 : 1.0;
    Complex* a = data.data() + offset;
    Complex* b = a + stride;

    for (std::size_t k = 0; k < stride; ++k) {
        const double angle = sign * M_PI * static_cast<double>(k) / static_cast<double>(stride);
        const ISA::Pack w  = ISA::setc(std::cos(angle), std::sin(angle));
        const ISA::Pack u  = ISA::load(a[k]);
        const ISA::Pack v  = ISA::cmul(w, ISA::load(b[k]));
        ISA::store(a[k], ISA::add(u, v));
        ISA::store(b[k], ISA::sub(u, v));
    }
}

void Radix4::butterfly(Buffer& data, std::size_t /*n*/,
                       std::size_t offset, std::size_t stride,
                       Direction dir) const {
    const double sign = (dir == Direction::Forward) ? -1.0 : 1.0;
    // TODO: same ISA, four streams + mul_j
    (void)data; (void)offset; (void)stride; (void)sign;
}

}  // namespace FFTop::CPU
