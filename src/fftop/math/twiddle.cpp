#include "fftop/math/twiddle.h"
#include <cmath>

namespace FFTop {
static constexpr Real K_TWO_PI = Real(6.283185307179586476925286766559L);

std::shared_ptr<const TwiddleTable> TwiddleCache::get(std::size_t n) {
    // Empty twiddle table
    if (n == 0) {
        static const auto empty = std::make_shared<const TwiddleTable>();
        return empty;
    }

    // Check cache
    {
        std::lock_guard<std::mutex> lock(mu_); // scoped lock
        auto it = entries_.find(n);
        if (it != entries_.end()) {
            // cache hit
            return it->second;
        }
    }

    // Create twiddle table
    auto table = std::make_shared<TwiddleTable>(n);
    for (std::size_t k = 0; k < n; ++k) {
        const Real angle = -K_TWO_PI * Real(k) / Real(n);
        (*table)[k]      = {std::cos(angle), std::sin(angle)};
    }

    // Insert into cache
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = entries_.try_emplace(n, std::move(table));
    
    // return table
    return it->second;
}

TwiddleCache& default_twiddles() {
    static TwiddleCache instance;
    return instance;
}

}  // namespace FFTop
