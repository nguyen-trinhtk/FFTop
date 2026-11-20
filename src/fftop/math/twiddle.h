#pragma once

#include "fftop/types.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace FFTop {
constexpr size_t MAX_RADIX = 32; // at most radix-32 (which is very unlikely)

// Forward twiddles table for a pair (n, radix)
// stored as one contiguous block per stage,
// where stages are increasing (0...logN - 1)
// Note: use conjugate for inverse FFT
struct Twiddles {
    std::size_t n = 0;
    std::size_t radix = 0;
    std::vector<Complex> w;

    // Block for the stage with stride s: (radix - 1) * s entries
    const Complex* stage(std::size_t s) const { return w.data() + (s - 1); }
};

// Global cache keyed by (n, radix)
class TwiddleCache {
public:
    std::shared_ptr<const Twiddles> get(std::size_t n, std::size_t radix);
private:
    struct KeyHash {
        std::size_t operator()(const std::pair<std::size_t, std::size_t>& k) const noexcept {
            return k.first * (MAX_RADIX + 1) + k.second; // no collision
        }
    };
    std::mutex mu_;

    // Hash map
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::shared_ptr<const Twiddles>, KeyHash> entries_;
};

// Singleton default twiddle cache
TwiddleCache& default_twiddles();
}  // namespace FFTop
