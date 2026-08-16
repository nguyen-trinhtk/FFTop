#pragma once

#include "fftop/types.h"
#include "fftop/plan/plan.h"

#include <cstddef>
#include <optional>
#include <unordered_map>

namespace FFTop {

struct CacheKey {
    std::size_t size;
    Backend     backend;
    Direction   direction;

    bool operator==(const CacheKey& o) const {
        return size == o.size && backend == o.backend && direction == o.direction;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const noexcept {
        std::size_t h = k.size;
        h ^= static_cast<std::size_t>(k.backend)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>(k.direction) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

class PlanCache {
public:
    std::optional<FFTPlan> find(std::size_t size, const FFTOptions& options) const;
    void store(const FFTPlan& plan);
    void clear();

private:
    std::unordered_map<CacheKey, FFTPlan, CacheKeyHash> entries_;
};

}  // namespace FFTop
