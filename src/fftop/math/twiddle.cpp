#include "fftop/math/twiddle.h"
#include <cmath>
#include <stdexcept>

namespace FFTop {
namespace {
constexpr Real TWO_PI = Real(6.283185307179586476925286766559L);
Complex cis_neg(std::size_t num, std::size_t den) {
    // cis(-2pi * num / den), with cis = cos - i * sin
    const Real angle = -TWO_PI * Real(num) / Real(den);
    return {std::cos(angle), std::sin(angle)};
}
std::vector<Complex> build_stages(std::size_t n, std::size_t radix) {
    // Each stage i with stride s = (radix)^i will holds s twiddles
    // and is preceded by s - 1 twiddles
    std::vector<Complex> w;
    w.reserve(n > 0 ? n - 1 : 0);
    // for each stride (~each stage)
    for (std::size_t s = 1; s < n; s *= radix) {
        // for each input in a radix-B block
        for (std::size_t m = 1; m < radix; ++m) {  // m=0 doesnt need twiddle
            // for each butterfly offset in stride
            for (std::size_t k = 0; k < s; ++k) {
                w.push_back(cis_neg(m * k, radix * s));
            }
        }
    }
    return w;
}
}  // namespace

Complex root_of_unity(std::size_t k, std::size_t n) { return cis_neg(k, n); }

Twiddles make_twiddles(std::size_t n, std::size_t radix) {
    if (radix < 2 || radix > MAX_RADIX)
        throw std::invalid_argument("make_twiddles: radix must be in [2, MAX_RADIX]");
    return Twiddles{n, radix, build_stages(n, radix)};
}
}  // namespace FFTop
