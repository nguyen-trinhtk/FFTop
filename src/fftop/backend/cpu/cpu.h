#pragma once

#include "fftop/backend/ibackend.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/traversal_strategy.h"
#include "fftop/backend/cpu/execution_mode.h"

#include <memory>
#include <string>

namespace FFTop {

class CPUBackend final : public IBackend {
public:
    CPUBackend(std::unique_ptr<CPU::IRadixB>         butterfly,
               std::unique_ptr<CPU::ITraversalStrategy> traversal_strategy,
               std::unique_ptr<CPU::IExecutionMode>     execution_mode);

    bool        is_available() const override { return true; }
    std::string name()         const override { return "CPU"; }
    void        execute(const FFTPlan& plan, const Buffer& input, Buffer& output) override;

private:
    std::unique_ptr<CPU::IRadixB>         butterfly_;
    std::unique_ptr<CPU::ITraversalStrategy> traversal_strategy_;
    std::unique_ptr<CPU::IExecutionMode>     execution_mode_;
};

}  // namespace FFTop
