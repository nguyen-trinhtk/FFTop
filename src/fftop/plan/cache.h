#pragma once

#include "fftop/main/types.h"
#include "fftop/plan/plan.h"

#include <optional>

namespace FFTop {

class PlanCache {
public:
    std::optional<FFTPlan> find(std::size_t size, const FFTOptions& options) const;
    void store(const FFTPlan& plan);
    void clear();
};

}  // namespace FFTop
