#pragma once

#include "fftop/backend/cpu/cpu.h"
#include "fftop/backend/gpu/naive.h"
#include "fftop/plan/plan.h"
#include "fftop/system/config.h"

#include <memory>
#include <vector>

namespace FFTop {

inline std::unique_ptr<IBackend> make_backend(const FFTPlan& plan, const SystemConfig& sys) {
    if (plan.backend == Backend::GPU || (plan.backend == Backend::Auto && sys.has_gpu)) {
        auto gpu = std::make_unique<NaiveGPUBackend>();
        if (gpu->is_available()) return gpu;
    }
    return std::make_unique<CPUBackend>();
}

inline std::vector<std::unique_ptr<IBackend>> all_backends() {
    std::vector<std::unique_ptr<IBackend>> out;
    out.push_back(std::make_unique<CPUBackend>());
    out.push_back(std::make_unique<NaiveGPUBackend>());
    return out;
}

}  // namespace FFTop
