#pragma once

#include "fftop/plan/cache.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#include <cstddef>

namespace FFTop {

class Planner {
public:
    explicit Planner(SystemConfig system, PlanCache* cache = nullptr);
    FFTPlan make_plan(std::size_t size, const FFTOptions& options);

private:
    SystemConfig system_;
    PlanCache*   cache_;
};

}  // namespace FFTop
