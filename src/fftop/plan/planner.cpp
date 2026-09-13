#include "fftop/plan/planner.h"

#include "fftop/math/fft_math.h"

namespace FFTop {
Planner::Planner(SystemConfig system, PlanCache* cache)
    : system_(system), cache_(cache) {}

FFTPlan Planner::make_plan(std::size_t size, const FFTOptions& options) {
    if (this->cache_) {
        auto hit = this->cache_->find(size, options);
        if (hit) {
            return *hit;
        }
    }

    FFTPlan plan;
    plan.size      = size;
    plan.direction = options.direction;
    plan.radix     = Math::is_power_of(size, 4) ? RadixPolicy::Radix4 : RadixPolicy::Radix2;

    if (options.hardware_target == HardwareTarget::Auto) {
        plan.hardware_target =
            this->system_.cuda ? HardwareTarget::GPU : HardwareTarget::CPU;
    } else {
        plan.hardware_target = options.hardware_target;
    }

    if (this->cache_) {
        this->cache_->store(plan);
    }
    return plan;
}
}  // namespace FFTop
