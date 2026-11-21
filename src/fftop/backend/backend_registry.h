#pragma once

#include "fftop/backend/gpu/gpu_options.h"
#include "fftop/backend/ibackend.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#include <memory>

namespace FFTop {
std::unique_ptr<IBackend> make_backend(const FFTPlan& plan,
        const SystemConfig& sys = system_config(),
        CPUPlanOptions cpu_options = {}, GPUPlanOptions gpu_options = {});
}  // namespace FFTop
