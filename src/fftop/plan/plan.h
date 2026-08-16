#pragma once

#include "fftop/main/types.h"

#include <cstddef>

namespace FFTop {

enum class Decomposition { Direct, FourStep };
enum class Kernel        { Radix2, Radix3, MixedRadix };

struct FFTPlan {
    std::size_t   size          = 0;
    Backend       backend       = Backend::Auto;
    Direction     direction     = Direction::Forward;
    Decomposition decomposition = Decomposition::Direct;
    Kernel        kernel        = Kernel::Radix2;
    bool          parallel      = false;
};

}  // namespace FFTop
