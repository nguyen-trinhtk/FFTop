#pragma once

#include "fftop/types.h"

#include <cmath>

namespace FFTop::Ref {

inline Buffer dft(const Buffer& input, Direction direction = Direction::Forward) {
    const std::size_t n = input.size();
    Buffer output(n);
    if (n == 0) return output;

    const Real sign  = direction == Direction::Forward ? Real(-1) : Real(1);
    const Real twopi = Real(2) * std::acos(Real(-1));

    for (std::size_t k = 0; k < n; ++k) {
        Complex sum{};
        for (std::size_t t = 0; t < n; ++t) {
            const Real angle = sign * twopi * Real(k) * Real(t) / Real(n);
            sum += input[t] * Complex(std::cos(angle), std::sin(angle));
        }
        if (direction == Direction::Inverse) sum /= Real(n);
        output[k] = sum;
    }
    return output;
}

}  // namespace FFTop::Ref
