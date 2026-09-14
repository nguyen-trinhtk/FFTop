#pragma once

#include "fftop/backend/ibackend.h"

namespace FFTop {

// Abstract base for GPU backends. Owns the is_available() check so
// concrete subclasses don't repeat it.
class GPUBackend : public IBackend {
public:
    bool is_available() const override;
};
}  // namespace FFTop
