#include "fftop/math/twiddle.h"
#include <cmath>
#include <stdexcept>

namespace FFTop {
namespace {
// File-private helpers
constexpr Real K_TWO_PI = Real(6.283185307179586476925286766559L);

Complex cis_neg(std::size_t num, std::size_t den) {
    // cis(-2pi * num / den)
    // with cis(theta) = cos(theta) - i * sin(theta))
    const Real angle = -K_TWO_PI * Real(num) / Real(den);
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
        for (std::size_t m = 1; m < radix; ++m) { // m=0 doesnt need twiddle
            // for each butterfly offset in stride
            for (std::size_t k = 0; k < s; ++k) {
                w.push_back(cis_neg(m * k, radix * s));
            }
        }
    }
    return w;
}
}  // namespace

// Twiddle cache class methods
std::shared_ptr<const Twiddles> TwiddleCache::get(std::size_t n, std::size_t radix) {
    if (radix < 2 || radix > MAX_RADIX)
        throw std::invalid_argument("TwiddleCache::get: radix must be in [2, MAX_RADIX]");
    const std::pair<std::size_t, std::size_t> key{n, radix};
    // Check existing twiddle in cache
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = entries_.find(key);
        if (it != entries_.end()) // cache hit
            return it->second;
    }
    // Build twiddle table
    auto table   = std::make_shared<Twiddles>();
    table->n     = n;
    table->radix = radix;
    table->w     = build_stages(n, radix);

    // Insert into cache (first writer wins)
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = entries_.try_emplace(key, std::move(table));
    return it->second;
}

// Singleton
TwiddleCache& default_twiddles() {
    static TwiddleCache instance;
    return instance;
}
}  // namespace FFTop
