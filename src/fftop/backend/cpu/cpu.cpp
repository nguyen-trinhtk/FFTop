#include "fftop/backend/cpu/cpu.h"

namespace FFTop {

CPUBackend::CPUBackend(std::unique_ptr<CPU::IRadixB>         butterfly,
                       std::unique_ptr<CPU::ITraversalStrategy> traversal_strategy,
                       bool                                     parallel)
    : butterfly_(std::move(butterfly))
    , traversal_strategy_(std::move(traversal_strategy))
    , parallel_(parallel) {}

void CPUBackend::execute(const FFTPlan& plan, const Buffer& input, Buffer& output) {
    output = input;
    const auto T = plan_twiddles(plan, butterfly_->radix());
    traversal_strategy_->run(output, plan.size, plan.direction, *butterfly_, parallel_,
                             *T);
}
}  // namespace FFTop
