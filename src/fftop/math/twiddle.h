#pragma once

#include "fftop/types.h"

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace FFTop {
constexpr size_t MAX_RADIX = 32;  // at most radix-32 (which is very unlikely)

// Forward twiddles table for a (n, radix) set
// Note: use conjugate for inverse FFT
struct Twiddles {
    std::size_t n     = 0;
    std::size_t radix = 0;
    std::vector<Complex> w;  // twiddles stored stage-by-stage

    // Ptr to buffer to stage with stride s
    const Complex* stage(std::size_t s) const { return w.data() + (s - 1); }
};

// W_n^k = e^{-2 pi i k / n}
Complex root_of_unity(std::size_t k, std::size_t n);

// Twiddle table (on host)
Twiddles make_twiddles(std::size_t n, std::size_t radix);

// Twiddle table cache
// each backend owns one table cache
template <class Key, class Table>
class TableCache {
public:
    template <class Build>
    std::shared_ptr<const Table> get(const Key& key, Build&& build) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = entries_.find(key);
            if (it != entries_.end()) return it->second;  // cache hit
        }
        std::shared_ptr<const Table> table = build();  // build outside the lock
        std::lock_guard<std::mutex> lock(mu_);
        return entries_.try_emplace(key, std::move(table)).first->second;  // first writer wins
    }

private:
    std::mutex mu_;
    std::map<Key, std::shared_ptr<const Table>> entries_;
};
}  // namespace FFTop
