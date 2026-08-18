#include "fftop/backend/cpu/simd/simd.h"

#include <cmath>

namespace FFTop::CPU {

void radix2_dit_scalar(Complex* data, std::size_t offset, std::size_t stride, double sign) {
    Complex* a = data + offset;
    Complex* b = data + offset + stride;

    for (std::size_t k = 0; k < stride; ++k) {
        const double  angle   = sign * M_PI * static_cast<double>(k) / static_cast<double>(stride);
        const Complex twiddle = {std::cos(angle), std::sin(angle)};
        const Complex u       = a[k];
        const Complex v       = twiddle * b[k];
        a[k] = u + v;
        b[k] = u - v;
    }
}

}  // namespace FFTop::CPU
