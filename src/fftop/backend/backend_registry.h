#pragma once

#include "fftop/backend/cpu/cpu.h"
#include "fftop/backend/gpu/naive.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#include <memory>
#include <vector>

namespace FFTop {

inline std::unique_ptr<CPUBackend> make_cpu_backend(const FFTPlan& plan, Simd simd) {
    auto butterfly = [&]() -> std::unique_ptr<CPU::IRadixB> {
        if (plan.radix == RadixPolicy::Radix4)
            return std::make_unique<CPU::Radix4>(simd);
        return std::make_unique<CPU::Radix2>(simd);  // Radix3/MixedRadix not implemented
    }();

    auto traversal_strategy = [&]() -> std::unique_ptr<CPU::ITraversalStrategy> {
        if (plan.traversal == Traversal::Recursive)
            return std::make_unique<CPU::RecursiveTraversalStrategy>();
        return std::make_unique<CPU::IterativeTraversalStrategy>();
    }();

    auto execution_mode = [&]() -> std::unique_ptr<CPU::IExecutionMode> {
        if (plan.execution == Execution::Parallel)
            return std::make_unique<CPU::ParallelExecutionMode>();
        return std::make_unique<CPU::SerialExecutionMode>();
    }();

    return std::make_unique<CPUBackend>(
        std::move(butterfly), std::move(traversal_strategy), std::move(execution_mode));
}

inline std::unique_ptr<CPUBackend> make_cpu_backend(const FFTPlan& plan) {
    return make_cpu_backend(plan, system_config().kernel_simd);
}

inline std::unique_ptr<IBackend> make_backend(const FFTPlan& plan) {
    if (plan.backend == Backend::GPU) {
        auto gpu = std::make_unique<NaiveGPUBackend>();
        if (gpu->is_available()) return gpu;
    }
    return make_cpu_backend(plan);
}

inline std::vector<std::unique_ptr<IBackend>> all_backends() {
    std::vector<std::unique_ptr<IBackend>> out;
    out.push_back(make_cpu_backend(FFTPlan{}));
    out.push_back(std::make_unique<NaiveGPUBackend>());
    return out;
}

}  // namespace FFTop
