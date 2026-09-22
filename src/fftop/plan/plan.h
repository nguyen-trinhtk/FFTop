#pragma once

#include "fftop/types.h"

#include <cstddef>
#include <optional>

namespace FFTop {

enum class RadixPolicy { Radix2, Radix4 };
enum class GPUKernelStrategy { CooleyTukeyGlobal, StockhamGlobal, StockhamShared };
// Shared by all backends (cached by the planner).
struct FFTPlan {
    std::size_t    size            = 0;
    HardwareTarget hardware_target = HardwareTarget::Auto;
    Direction      direction       = Direction::Forward;
    RadixPolicy    radix           = RadixPolicy::Radix2;
    GPUKernelStrategy gpu_strategy = GPUKernelStrategy::CooleyTukeyGlobal;
};

// CPU backend construction only; not part of the cached plan.
enum class Traversal { Recursive, Iterative };
enum class Execution { Serial, Parallel };

struct CPUPlanOptions {
    Traversal                 traversal = Traversal::Iterative;
    std::optional<Execution> execution;  // unset => Parallel if sys.openmp, else Serial
};

}  // namespace FFTop
