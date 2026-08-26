#pragma once

// Global-memory-only Cooley-Tukey FFT backend.
// Parallel to CPUBackend: composed of a radix policy and a traversal strategy.
// No execution-mode layer — the CUDA thread grid provides parallelism.

#include "fftop/backend/gpu/gpu_backend.h"
#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"

#include <memory>
#include <string>

namespace FFTop {

class CooleyTukeyGPUBackend final : public GPUBackend {
public:
    CooleyTukeyGPUBackend(std::unique_ptr<GPU::IGPURadix>             radix,
                          std::unique_ptr<GPU::IGPUTraversalStrategy> traversal);

    std::string name() const override { return "GPU/CooleyTukey"; }
    void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) override;

private:
    std::unique_ptr<GPU::IGPURadix>             radix_;
    std::unique_ptr<GPU::IGPUTraversalStrategy> traversal_;
};

}  // namespace FFTop
