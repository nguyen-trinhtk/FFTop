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
                                     const IRadixB& butterfly, bool parallel,
                                     const Twiddles& T) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    recurse(data, n, 0, dir, butterfly, parallel, T);
}

void RecursiveTraversalStrategy::recurse(Buffer& data, std::size_t n, std::size_t offset,
                                         Direction dir, const IRadixB& butterfly,
                                         bool parallel,
                                         const Twiddles& T) const {
    if (n <= 1) return;

    const std::size_t r   = butterfly.radix();
    const std::size_t sub = n / r;

    // even and odd partitioning
    const bool par = use_parallel(parallel, r, sub);
    FFTOP_OMP_FOR_IF(par)
    for (omp_index branch = 0; branch < static_cast<omp_index>(r); ++branch) {
        recurse(data, sub, offset + static_cast<std::size_t>(branch) * sub, dir, butterfly,
                parallel, T);
    }
    butterfly.butterfly(data, offset, sub, dir, T.stage(sub));
}

void IterativeTraversalStrategy::run(Buffer& data, std::size_t n, Direction dir,
                                     const IRadixB& butterfly, bool parallel,
                                     const Twiddles& T) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    const std::size_t r = butterfly.radix();
    // for each stage, process butterflies
    for (std::size_t stride = 1; stride < n; stride *= r) {
        const std::size_t group_size = stride * r;
        const std::size_t num_groups = n / group_size;
        const Complex*    Ws         = T.stage(stride);

        const bool par = use_parallel(parallel, num_groups, stride);
        FFTOP_OMP_FOR_IF(par)
        for (omp_index g = 0; g < static_cast<omp_index>(num_groups); ++g) {
            butterfly.butterfly(data, static_cast<std::size_t>(g) * group_size, stride, dir, Ws);
        }
    }
}
}  // namespace FFTop::CPU
