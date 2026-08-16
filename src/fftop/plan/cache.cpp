#include "fftop/plan/cache.h"

namespace FFTop {

std::optional<FFTPlan> PlanCache::find(std::size_t size, const FFTOptions& options) const {
    auto it = entries_.find({size, options.backend, options.direction});
    if (it == entries_.end()) return std::nullopt;
    return it->second;
}

void PlanCache::store(const FFTPlan& plan) {
    entries_[{plan.size, plan.backend, plan.direction}] = plan;
}

void PlanCache::clear() {
    entries_.clear();
}

}  // namespace FFTop
