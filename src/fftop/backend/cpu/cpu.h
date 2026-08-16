#pragma once

#include "fftop/backend/ibackend.h"

namespace FFTop {

class CPUBackend final : public IBackend {
public:
    bool is_available() const override { return true; }
    std::string name() const override { return "CPU"; }
    void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) override;
};

}  // namespace FFTop
