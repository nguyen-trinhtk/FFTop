#include "fftop/main/fft.h"

#include "fftop/backend/cpu/cpu.h"
#include "fftop/backend/gpu/naive.h"
#include "fftop/plan/cache.h"
#include "fftop/plan/planner.h"
#include "fftop/system/config.h"

#include <memory>

namespace FFTop {

namespace {

PlanCache& default_cache() {
    static PlanCache instance;
    return instance;
}

SystemConfig& default_system() {
    static SystemConfig instance = detect_system_config();
    return instance;
}

std::unique_ptr<IBackend> select_backend(const FFTPlan& plan, const SystemConfig& sys) {
    if (plan.backend == Backend::GPU || (plan.backend == Backend::Auto && sys.has_gpu)) {
        auto gpu = std::make_unique<NaiveGPUBackend>();
        if (gpu->is_available()) return gpu;
    }
    return std::make_unique<CPUBackend>();
}

}  // namespace

Buffer fft(const Buffer& input, const FFTOptions& options) {
    SystemConfig& sys     = default_system();
    Planner       planner(sys, &default_cache());
    const FFTPlan plan    = planner.make_plan(input.size(), options);
    auto          backend = select_backend(plan, sys);
    Buffer        output;
    backend->execute(plan, input, output);
    return output;
}

}  // namespace FFTop
