#include "fftop/backend/cpu/traversal_strategy.h"

#include <cassert>
#include <utility>

namespace FFTop::CPU {

namespace {

// index with its base-`radix` digits reversed; radix 2 is plain bit reversal.
std::size_t digit_reverse(std::size_t index, std::size_t n, std::size_t radix) {
    std::size_t reversed = 0;
    for (std::size_t rest = n; rest > 1; rest /= radix) {
        reversed = reversed * radix + index % radix;
        index /= radix;
    }
    return reversed;
}

// DIT reads its sub-transforms as contiguous blocks, which only lines up once
// the input sits in digit-reversed order. Reversal is its own inverse, so
// swapping each i < reverse(i) pair permutes in place.
void digit_reverse_permute(Buffer& data, std::size_t n, std::size_t radix) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = digit_reverse(i, n, radix);
        if (i < j) std::swap(data[i], data[j]);
    }
}

}  // namespace

// ── Recursive ────────────────────────────────────────────────────────────────

void RecursiveTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode) const {
    if (n <= 1) return;
    assert(butterfly.supports(n) && "size must be a power of the radix");

    digit_reverse_permute(data, n, butterfly.radix());
    recurse(data, n, 0, dir, butterfly, execution_mode);
}

void RecursiveTraversalStrategy::recurse(Buffer& data, std::size_t n, std::size_t offset,
                                         Direction dir, const IRadixB& butterfly,
                                         const IExecutionMode& execution_mode) const {
    if (n <= 1) return;

    const std::size_t r   = butterfly.radix();
    const std::size_t sub = n / r;

    execution_mode.parallel_for(r, sub,
        [&](std::size_t branch) {
            recurse(data, sub, offset + branch * sub, dir, butterfly, execution_mode);
        });

    butterfly.butterfly(data, offset, sub, dir);
}

// ── Iterative ────────────────────────────────────────────────────────────────

void IterativeTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, const IExecutionMode& execution_mode) const {
    if (n <= 1) return;
    assert(butterfly.supports(n) && "size must be a power of the radix");

    digit_reverse_permute(data, n, butterfly.radix());

    const std::size_t r = butterfly.radix();
    for (std::size_t stride = 1; stride < n; stride *= r) {
        const std::size_t group_size = stride * r;
        const std::size_t num_groups = n / group_size;

        execution_mode.parallel_for(num_groups, stride, [&](std::size_t g) {
            butterfly.butterfly(data, g * group_size, stride, dir);
        });
    }
}

}  // namespace FFTop::CPU
