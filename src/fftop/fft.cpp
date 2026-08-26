#include "fftop/fft.h"

#include "fftop/backend/backend_registry.h"
#include "fftop/plan/cache.h"
#include "fftop/plan/planner.h"
#include "fftop/system.h"

namespace FFTop {
namespace {
PlanCache& default_cache() {
    static PlanCache instance;
    return instance;
}
}  // namespace

Buffer fft(const Buffer& input, const FFTOptions& options) {
    const SystemConfig& sys = system_config();
    Planner planner(sys, &default_cache());
    const FFTPlan plan = planner.make_plan(input.size(), options);
    auto backend = make_backend(plan);
    Buffer output;
    backend->execute(plan, input, output);
    return output;
}

}  // namespace FFTop
