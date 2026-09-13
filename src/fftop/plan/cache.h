#pragma once

#include "fftop/types.h"
#include "fftop/plan/plan.h"

#include <cstddef>
#include <optional>
#include <unordered_map>

static constexpr std::size_t FNV_PRIME = 0x9e3779b9;

namespace FFTop {

// Cache key
struct CacheKey {
    std::size_t size;
    HardwareTarget hardware_target;
    Direction   direction;

    bool operator==(const CacheKey& o) const {
        return size == o.size 
        && hardware_target == o.hardware_target 
        && direction == o.direction;
    }
};


// Key hash function: FNV-1a
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const noexcept {
        std::size_t h = k.size;
        h ^= static_cast<std::size_t>(k.hardware_target) + FNV_PRIME + (h << 6) + (h >> 2);
        h ^= static_cast<std::size_t>(k.direction) + FNV_PRIME + (h << 6) + (h >> 2);
        return h;
    }
};

// Unordered map of cache keys to plans
class PlanCache {
public:
    std::optional<FFTPlan> find(std::size_t size, const FFTOptions& options) const;
    void store(const FFTPlan& plan);
private:
    std::unordered_map<CacheKey, FFTPlan, CacheKeyHash> entries_;
};

}  // namespace FFTop
