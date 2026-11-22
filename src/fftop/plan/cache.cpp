#include "fftop/plan/cache.h"

#include <utility>

namespace FFTop {
std::shared_ptr<const Plan> PlanCache::find(std::size_t size, HardwareTarget target) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = entries_.find({size, target});
    if (it == entries_.end()) return nullptr;
    return it->second;
}

std::shared_ptr<const Plan> PlanCache::store(std::size_t size, HardwareTarget target,
                                             std::shared_ptr<const Plan> plan) {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.try_emplace({size, target}, std::move(plan)).first->second;
}

void PlanCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    entries_.clear();
}

std::size_t PlanCache::entries() const {
    std::lock_guard<std::mutex> lock(mu_);
    return entries_.size();
}
}  // namespace FFTop
