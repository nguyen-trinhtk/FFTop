#include "fftop/fft.h"

#include "fftop/backend/backend_registry.h"
#include "fftop/plan/cache.h"
#include "fftop/plan/planner.h"
#include "fftop/system.h"

namespace FFTop {
namespace {
// Singleton plan cache
PlanCache& default_cache() {
    static PlanCache instance;
    return instance;
}
}  // namespace

Buffer fft(const Buffer& input, const FFTOptions& options) {
    // check system config
    const SystemConfig& sys = system_config(); 

    // create planner
    Planner planner(sys, &default_cache());

    // make plan based on input size
    const FFTPlan plan = planner.make_plan(input.size(), options);
    
    // make backend
    auto backend = make_backend(plan, sys);

    // execute plan
    Buffer output;
    backend->execute(plan, input, output);
    return output;
}

}  // namespace FFTop
