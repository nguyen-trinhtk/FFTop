#pragma once

#include "fftop/main/types.h"
#include "fftop/plan/plan.h"

#include <string>

namespace FFTop {

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual bool is_available() const = 0;
    virtual std::string name()         const = 0;
    virtual void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) = 0;
};

}  // namespace FFTop
