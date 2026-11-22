#include "fftop/fft.h"

#include "fftop/plan/cache.h"
#include "fftop/plan/planner.h"
#include "fftop/system.h"

namespace FFTop {
namespace {
// Process-wide planner and its cache of plans
const Planner& default_planner() {
    static PlanCache cache;
    static const Planner planner(system_config(), EstimateParams{}, &cache);
    return planner;
}
}  // namespace

// Find or build the plan, then execute it
Buffer fft(const Buffer& input, const FFTOptions& options) {
    Buffer output;
    default_planner()
        .plan(input.size(), options.hardware_target)
        ->execute(output, input, options.direction);
    return output;
}

// On-device: for GPU FFT when data is already on GPU
void fft_device(Complex* d_out, const Complex* d_in, std::size_t n, Direction direction) {
    default_planner().plan(n, HardwareTarget::GPU)->execute_device(d_out, d_in, direction);
}
}  // namespace FFTop
