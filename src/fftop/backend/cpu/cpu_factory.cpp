#include "fftop/backend/cpu/cpu_factory.h"
#include "fftop/backend/cpu/cpu.h"
#include <memory>

namespace FFTop {
namespace {
std::unique_ptr<CPU::IRadixB> make_cpu_radix(RadixPolicy radix, SIMD simd) {
    switch (radix) {
        case RadixPolicy::Radix4:
            return std::make_unique<CPU::Radix4>(simd);
        case RadixPolicy::Radix2:
            return std::make_unique<CPU::Radix2>(simd);
    }
    return std::make_unique<CPU::Radix2>(simd);
}

std::unique_ptr<CPU::ITraversalStrategy> make_cpu_traversal(Traversal traversal) {
    switch (traversal) {
        case Traversal::Recursive:
            return std::make_unique<CPU::RecursiveTraversalStrategy>();
        case Traversal::Iterative:
            return std::make_unique<CPU::IterativeTraversalStrategy>();
    }
    return std::make_unique<CPU::IterativeTraversalStrategy>();
}

std::unique_ptr<CPU::IExecutionMode> make_cpu_execution(Execution execution) {
    switch (execution) {
        case Execution::Parallel:
            return std::make_unique<CPU::ParallelExecutionMode>();
        case Execution::Serial:
            return std::make_unique<CPU::SerialExecutionMode>();
    }
    return std::make_unique<CPU::SerialExecutionMode>();
}

Execution resolve_execution(const CPUPlanOptions& opts, const SystemConfig& sys) {
    if (opts.execution) return *opts.execution;
    return sys.openmp ? Execution::Parallel : Execution::Serial;
}
}  // namespace

std::unique_ptr<IBackend> make_cpu_backend(const FFTPlan& plan,
                                           const SystemConfig& sys,
                                           CPUPlanOptions opts) {
    return std::make_unique<CPUBackend>(
        make_cpu_radix(plan.radix, sys.simd),
        make_cpu_traversal(opts.traversal),
        make_cpu_execution(resolve_execution(opts, sys)));
}
}  // namespace FFTop
