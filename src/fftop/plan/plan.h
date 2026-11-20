#pragma once

#include "fftop/math/twiddle.h"
#include "fftop/types.h"

#include <cstddef>
#include <memory>
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

    // Twiddles built at plan time, in the layout the target backend reads:
    // Per-stage table for the plan radix (CPU and GPU). Null for hand-built plans.
    std::shared_ptr<const Twiddles> twiddles;
};

// Per-stage twiddles for the radix a backend needs. Returns the plan's
// table when it matches, otherwise falls back to the cache (hand-built plans).
inline std::shared_ptr<const Twiddles> plan_twiddles(const FFTPlan& plan, std::size_t radix) {
    const auto& t = plan.twiddles;
    if (t && t->n == plan.size && t->radix == radix) 
        return t;
    return default_twiddles().get(plan.size, radix);
}

// CPU backend construction only; not part of the cached plan.
enum class Traversal { Recursive, Iterative };
enum class Execution { Serial, Parallel };

struct CPUPlanOptions {
    Traversal                 traversal = Traversal::Iterative;
    std::optional<Execution> execution;  // unset => Parallel if sys.openmp, else Serial
};

}  // namespace FFTop
