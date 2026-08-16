#include "fftop/plan/planner.h"

namespace FFTop {

Planner::Planner(SystemConfig system, PlanCache* cache)
    : system_(system), cache_(cache) {}

FFTPlan Planner::make_plan(std::size_t size, const FFTOptions& options) {
    if (this->cache_) { 
        if (auto hit = this->cache_->find(size, options)) {
            return *hit;
        }
    }
    FFTPlan plan;
    plan.size  = size;
    plan.backend   = (options.backend == Backend::Auto)
                         ? (this->system_.has_gpu ? Backend::GPU : Backend::CPU)
                         : options.backend;
    plan.direction = options.direction;
    plan.parallel  = this->system_.enable_parallelism && this->system_.cpu_threads > 1;
    if (this->cache_) {
        this->cache_->store(plan);
    }
    return plan;
}

}  // namespace FFTop
