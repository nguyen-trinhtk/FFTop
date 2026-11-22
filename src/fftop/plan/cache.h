#pragma once

#include "fftop/plan/plan.h"
#include "fftop/types.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace FFTop {
inline constexpr std::size_t FNV_PRIME = 0x9e3779b9;

// Key on (size, HW target)
struct CacheKey {
    std::size_t size;
    HardwareTarget target;  // CPU/GPU/Auto
    bool operator==(const CacheKey& o) const { return size == o.size && target == o.target; }
};

// Key hash function: FNV-1a
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const noexcept {
        std::size_t h = k.size;
        h ^= static_cast<std::size_t>(k.target) + FNV_PRIME + (h << 6) + (h >> 2);
        return h;
    }
};

// Cache of (size, target) -> plan
// TODO: size limit (LRU) (using clear() for now)
class PlanCache {
public:
    std::shared_ptr<const Plan> find(std::size_t size, HardwareTarget target) const;
    std::shared_ptr<const Plan> store(std::size_t size, HardwareTarget target,
                                      std::shared_ptr<const Plan> plan);
    void clear();
    std::size_t entries() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<CacheKey, std::shared_ptr<const Plan>, CacheKeyHash> entries_;
};
}  // namespace FFTop
