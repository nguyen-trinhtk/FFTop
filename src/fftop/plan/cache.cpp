#include "fftop/plan/cache.h"

namespace FFTop {
std::optional<FFTPlan> PlanCache::find(std::size_t size, const FFTOptions& options) const {
    auto it = entries_.find({size, options.hardware_target, options.direction});
    if (it == entries_.end()) return std::nullopt;
    return it->second;
}

void PlanCache::store(const FFTPlan& plan) {
    entries_[{plan.size, plan.hardware_target, plan.direction}] = plan;
}
}  // namespace FFTop
