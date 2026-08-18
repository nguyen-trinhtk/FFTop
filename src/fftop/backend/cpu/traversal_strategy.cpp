#include "fftop/backend/cpu/traversal_strategy.h"

#include <utility>

namespace FFTop::CPU {

namespace {
void bit_reverse(Buffer& data, std::size_t n, int r) {
    if (r != 2 || n <= 2) return;

    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        while (bit != 0 && j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if (i < j) std::swap(data[i], data[j]);
    }
}
}  // namespace

// ── Recursive ────────────────────────────────────────────────────────────────

void RecursiveTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode) const {
    bit_reverse(data, n, butterfly.radix());
    recurse(data, n, 0, dir, butterfly, execution_mode);
}

void RecursiveTraversalStrategy::recurse(Buffer& data, std::size_t n, std::size_t offset,
                                         Direction dir, const IRadixB& butterfly,
                                         const IExecutionMode& execution_mode) const {
    if (n <= 1) return;

    const std::size_t sub = n / static_cast<std::size_t>(butterfly.radix());

    execution_mode.parallel_for(static_cast<std::size_t>(butterfly.radix()), sub,
        [&](std::size_t branch) {
            recurse(data, sub, offset + branch * sub, dir, butterfly, execution_mode);
        });

    butterfly.butterfly(data, n, offset, sub, dir);
}

// ── Iterative ────────────────────────────────────────────────────────────────

void IterativeTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode) const {
    if (n <= 1) return;

    bit_reverse(data, n, butterfly.radix());

    const auto r = static_cast<std::size_t>(butterfly.radix());
    for (std::size_t stride = 1; stride < n; stride *= r) {
        const std::size_t group_size = stride * r;
        const std::size_t num_groups = n / group_size;

        execution_mode.parallel_for(num_groups, stride, [&](std::size_t g) {
            butterfly.butterfly(data, group_size, g * group_size, stride, dir);
        });
    }
}

}  // namespace FFTop::CPU
