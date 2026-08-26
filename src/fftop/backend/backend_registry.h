#pragma once

#include "fftop/backend/cpu/cpu.h"
#include "fftop/backend/gpu/naive.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/cooley_tukey.h"
#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"
#endif

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

#if defined(FFTOP_ENABLE_CUDA)
inline std::unique_ptr<CooleyTukeyGPUBackend> make_cooley_tukey_gpu(const FFTPlan& plan) {
    auto radix = [&]() -> std::unique_ptr<GPU::IGPURadix> {
        if (plan.radix == RadixPolicy::Radix4)
            return std::make_unique<GPU::GPURadix4>();
        return std::make_unique<GPU::GPURadix2>();
    }();
    return std::make_unique<CooleyTukeyGPUBackend>(
        std::move(radix),
        std::make_unique<GPU::IterativeGPUTraversalStrategy>());
}
#endif

inline std::unique_ptr<IBackend> make_backend(const FFTPlan& plan) {
    if (plan.backend == Backend::GPU) {
#if defined(FFTOP_ENABLE_CUDA)
        auto gpu = make_cooley_tukey_gpu(plan);
        if (gpu->is_available()) return gpu;
#else
        auto gpu = std::make_unique<NaiveGPUBackend>();
        if (gpu->is_available()) return gpu;
#endif
    }
    return make_cpu_backend(plan);
}

inline std::vector<std::unique_ptr<IBackend>> all_backends() {
    std::vector<std::unique_ptr<IBackend>> out;
    out.push_back(make_cpu_backend(FFTPlan{}));
    out.push_back(std::make_unique<NaiveGPUBackend>());  // O(N²) DFT reference
#if defined(FFTOP_ENABLE_CUDA)
    out.push_back(make_cooley_tukey_gpu(FFTPlan{}));     // O(N log N) FFT
#endif
    return out;
}

}  // namespace FFTop
