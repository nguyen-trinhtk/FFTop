#include "fftop/backend/cpu/traversal_strategy.h"
#include "fftop/math/fft_math.h"

#include <cassert>
#include <utility>

namespace FFTop::CPU {

namespace {

// Rearrange input in digit-reversed order (DIT)
bool check_and_reorder_dit(Buffer& data, std::size_t n, const IRadixB& butterfly) {
    if (n <= 1) return false;
    assert(butterfly.supports(n) && "size must be a power of the radix");
    const std::size_t radix = butterfly.radix();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = Math::digit_reverse(i, n, radix);
        if (i < j) std::swap(data[i], data[j]);
    }
    return true;
}
}  // namespace

// Recursive
void RecursiveTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode,
                                     const Complex* W) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    recurse(data, n, 0, dir, butterfly, execution_mode, W, n);
}

void RecursiveTraversalStrategy::recurse(Buffer& data, std::size_t n, std::size_t offset,
                                         Direction dir, const IRadixB& butterfly,
                                         const IExecutionMode& execution_mode,
                                         const Complex* W, std::size_t n_full) const {
    if (n <= 1) return;

    const std::size_t r   = butterfly.radix();
    const std::size_t sub = n / r;

    // even and odd partitioning
    execution_mode.parallel_for(r, sub,
        [&](std::size_t branch) {
            recurse(data, sub, offset + branch * sub, dir, butterfly, execution_mode, W, n_full);
        });
    butterfly.butterfly(data, offset, sub, dir, W, n_full);
}

void IterativeTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode,
                                     const Complex* W) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    const std::size_t r = butterfly.radix();
    // for each stage, process butterflies
    for (std::size_t stride = 1; stride < n; stride *= r) {
        const std::size_t group_size = stride * r;
        const std::size_t num_groups = n / group_size;

        execution_mode.parallel_for(num_groups, stride, [&](std::size_t g) {
            butterfly.butterfly(data, g * group_size, stride, dir, W, n);
        });
    }
}
}  // namespace FFTop::CPU
