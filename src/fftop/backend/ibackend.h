#pragma once

#include "fftop/types.h"
#include "fftop/plan/plan.h"

#include <string>

namespace FFTop {
// Backend interface
// For now only CPU and NVIDIA GPU are supported.
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual bool is_available() const = 0;
    virtual std::string name() const = 0;
    virtual void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) = 0;
};
}  // namespace FFTop
