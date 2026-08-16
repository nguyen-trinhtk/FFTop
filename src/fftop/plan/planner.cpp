#include "fftop/plan/planner.h"

namespace FFTop {

Planner::Planner(SystemConfig system, PlanCache* cache)
    : system_(system), cache_(cache) {
}

FFTPlan Planner::make_plan(std::size_t size, const FFTOptions& options) {
    FFTPlan plan;
    plan.size      = size;
    plan.backend   = options.backend;
    plan.direction = options.direction;
    plan.parallel  = system_.enable_parallelism;
    return plan;
}

}  // namespace FFTop
