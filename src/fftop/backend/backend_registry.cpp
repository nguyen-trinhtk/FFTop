#include "fftop/backend/backend_registry.h"

#include "fftop/backend/cpu/cpu_factory.h"

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/gpu_factory.h"
#endif

#include <stdexcept>

namespace FFTop {

std::unique_ptr<IBackend> make_backend(const FFTPlan& plan,
                                       const SystemConfig& sys,
                                       CPUPlanOptions cpu) {
    const bool try_gpu = plan.hardware_target == HardwareTarget::GPU ||
                         (plan.hardware_target == HardwareTarget::Auto && sys.cuda);
    if (try_gpu) {
#if defined(FFTOP_ENABLE_CUDA)
        if (auto gpu = make_gpu_backend(plan)) return gpu;
#endif
        if (plan.hardware_target == HardwareTarget::GPU) {
            throw std::runtime_error("GPU backend requested but no CUDA device is available");
        }
    }
    return make_cpu_backend(plan, sys, cpu);
}

}  // namespace FFTop
