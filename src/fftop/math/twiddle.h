#pragma once

#include "fftop/types.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace FFTop {

// Forward twiddles: W[k] = cis(-2π k / n)
// Inverse is conj(W[k])
using TwiddleTable = std::vector<Complex>;

// Cache keyed by n, shared across backends and directions
class TwiddleCache {
public:
    std::shared_ptr<const TwiddleTable> get(std::size_t n);

private:
    std::mutex mu_;
    std::unordered_map<std::size_t, std::shared_ptr<const TwiddleTable>> entries_;
};

// Singleton default twiddle cache
TwiddleCache& default_twiddles();

// Get twiddle table from default cache
inline std::shared_ptr<const TwiddleTable> get_twiddles(std::size_t n) {
    return default_twiddles().get(n);
}
}  // namespace FFTop
