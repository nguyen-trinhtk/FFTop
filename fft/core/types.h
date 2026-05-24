#pragma once
#ifndef FFT_CORE_TYPES_H
#define FFT_CORE_TYPES_H
#include <complex>
#include <vector>
#include <functional>

namespace FFTCore {
    using Real = double;
    using Complex = std::complex<Real>;
    using FFTFunc = std::function<std::vector<Complex>(const std::vector<Complex>&)>;
}
#endif // FFT_CORE_TYPES_H