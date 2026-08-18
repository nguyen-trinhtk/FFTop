#include "fftop/backend/cpu/cpu.h"

namespace FFTop {

CPUBackend::CPUBackend(std::unique_ptr<CPU::IRadixB>         butterfly,
                       std::unique_ptr<CPU::ITraversalStrategy> traversal_strategy,
                       std::unique_ptr<CPU::IExecutionMode>     execution_mode)
    : butterfly_(std::move(butterfly))
    , traversal_strategy_(std::move(traversal_strategy))
    , execution_mode_(std::move(execution_mode)) {}

void CPUBackend::execute(const FFTPlan& plan, const Buffer& input, Buffer& output) {
    output = input;
    traversal_strategy_->run(output, plan.size, plan.direction, *butterfly_, *execution_mode_);
}

}  // namespace FFTop
