#pragma once

#include "fftop/backend/ibackend.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#include <memory>

namespace FFTop {
std::unique_ptr<IBackend> make_cpu_backend(const FFTPlan& plan,
                                           const SystemConfig& sys,
                                           CPUPlanOptions opts = {});
}  // namespace FFTop
