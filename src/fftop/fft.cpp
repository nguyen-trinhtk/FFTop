#include "fftop/fft.h"

#include "fftop/backend/backend_registry.h"
#include "fftop/plan/cache.h"
#include "fftop/plan/planner.h"
#include "fftop/system/config.h"

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
}  // namespace

Buffer fft(const Buffer& input, const FFTOptions& options) {
    SystemConfig& sys = default_system();
    Planner planner(sys, &default_cache());
    const FFTPlan plan    = planner.make_plan(input.size(), options);
    auto backend = make_backend(plan, sys);
    Buffer output;
    backend->execute(plan, input, output);
    return output;
}

}  // namespace FFTop
