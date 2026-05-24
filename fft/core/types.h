#pragma once
#ifndef FFT_CORE_TYPES_H
#define FFT_CORE_TYPES_H
#include <complex>
#include <vector>
#include <functional>
#include <cmath>

namespace FFTCore {
    using Real = double;
    using Complex = std::complex<Real>;
    using FFTFunc = std::function<void(const std::vector<Complex>&, std::vector<Complex>&)>;

    // Constant
    const Real PI = std::acos(-1.0);
}
#endif // FFT_CORE_TYPES_H