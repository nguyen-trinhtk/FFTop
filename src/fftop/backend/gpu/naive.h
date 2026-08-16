#pragma once

#include "fftop/backend/gpu/gpu_backend.h"

namespace FFTop {

class NaiveGPUBackend final : public GPUBackend {
public:
    std::string name()    const override { return "GPU/Naive"; }
    void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) override;
};

}  // namespace FFTop
