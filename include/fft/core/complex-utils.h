#pragma once
#include "fft/core/types.h"

namespace FFTCore {
    static inline Complex times_neg_i(Complex x) {
        return Complex(x.imag(), -x.real());
    }
} // namespace FFTCore
