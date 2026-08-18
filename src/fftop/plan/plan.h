#pragma once

#include "fftop/types.h"

#include <cstddef>

namespace FFTop {

enum class Decomposition { Direct, FourStep };
enum class RadixPolicy   { Radix2, Radix4, Radix3, MixedRadix };
enum class Traversal     { Recursive, Iterative };
enum class Execution     { Serial, Parallel };

struct FFTPlan {
    std::size_t   size          = 0;
    Backend       backend       = Backend::Auto;
    Direction     direction     = Direction::Forward;
    Decomposition decomposition = Decomposition::Direct;
    RadixPolicy   radix         = RadixPolicy::Radix2;
    Traversal     traversal     = Traversal::Iterative;
    Execution     execution     = Execution::Serial;
};

}  // namespace FFTop
