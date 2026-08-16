#include "fftop/plan/cache.h"

namespace FFTop {

std::optional<FFTPlan> PlanCache::find(std::size_t, const FFTOptions&) const {
    return std::nullopt;
}

void PlanCache::store(const FFTPlan&) {
}

void PlanCache::clear() {
}

}  // namespace FFTop
