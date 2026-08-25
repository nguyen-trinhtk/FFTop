#include "fftop/plan/planner.h"

#include "fftop/math/integer.h"

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
    plan.size      = size;
    plan.backend   = (options.backend == Backend::Auto)
                         ? (this->system_.has_gpu ? Backend::GPU : Backend::CPU)
                         : options.backend;
    plan.direction = options.direction;
    // Radix-4 halves the number of stages; it only tiles sizes that are 4^p.
    plan.radix     = is_power_of(size, 4) ? RadixPolicy::Radix4 : RadixPolicy::Radix2;
    plan.traversal = Traversal::Iterative;      // TODO: expose via FFTOptions if needed
    plan.execution = (this->system_.enable_parallelism && this->system_.cpu_threads > 1)
                         ? Execution::Parallel
                         : Execution::Serial;
    if (this->cache_) {
        this->cache_->store(plan);
    }
    return plan;
}

}  // namespace FFTop
