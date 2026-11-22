#include "fftop/backend/cpu/traversal.h"
#include "fftop/math/fft_math.h"

#include <cassert>
#include <utility>

namespace FFTop::CPU {

namespace {

// Rearrange input in digit-reversed order (DIT)
bool check_and_reorder_dit(Complex* data, std::size_t n, const IRadixR& butterfly) {
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
void RecursiveTraversalStrategy::run(Complex* data, std::size_t n, Direction dir,
                                     const IRadixR& butterfly, int threads,
                                     const Twiddles& T) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    recurse(data, n, dir, butterfly, threads, T);
}

void RecursiveTraversalStrategy::recurse(Complex* data, std::size_t n, Direction dir,
                                         const IRadixR& butterfly, int threads,
                                         const Twiddles& T) const {
    if (n <= 1) return;

    const std::size_t r   = butterfly.radix();
    const std::size_t sub = n / r;

    // even and odd partitioning
    const bool par = use_parallel(threads, r);
    FFTOP_OMP_FOR_IF(par, threads)
    for (omp_index branch = 0; branch < static_cast<omp_index>(r); ++branch) {
        recurse(data + static_cast<std::size_t>(branch) * sub, sub, dir, butterfly, threads, T);
    }
    butterfly.butterfly(data, sub, dir, T.stage(sub));
}

void IterativeTraversalStrategy::run(Complex* data, std::size_t n, Direction dir,
                                     const IRadixR& butterfly, int threads,
                                     const Twiddles& T) const {
    if (!check_and_reorder_dit(data, n, butterfly)) return;
    const std::size_t r = butterfly.radix();
    // for each stage, process butterflies
    for (std::size_t stride = 1; stride < n; stride *= r) {
        const std::size_t group_size = stride * r;
        const std::size_t num_groups = n / group_size;
        const Complex* Ws            = T.stage(stride);

        const bool par = use_parallel(threads, num_groups);
        FFTOP_OMP_FOR_IF(par, threads)
        for (omp_index g = 0; g < static_cast<omp_index>(num_groups); ++g) {
            butterfly.butterfly(data + static_cast<std::size_t>(g) * group_size, stride, dir, Ws);
        }
    }
}
}  // namespace FFTop::CPU
