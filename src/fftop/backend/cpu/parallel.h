#pragma once

#include <cstddef>

namespace FFTop::CPU {
inline constexpr std::size_t PARALLEL_GRAIN = 1024;
// (Signed) loop index for FFTOP_OMP_FOR_IF loops
using omp_index = std::ptrdiff_t;
// check if loop worth parallelizing (>= grain size)
inline bool use_parallel(bool parallel, std::size_t n, std::size_t work_per_item) {
    return parallel && n > 1 && n * work_per_item >= PARALLEL_GRAIN;
}
}  // namespace FFTop::CPU

// Put this directive before a for loop that can possibly 
// be parallelized when OpenMP is enabled and cond is true
#define FFTOP_PRAGMA(x) _Pragma(#x)
#ifdef FFTOP_ENABLE_OPENMP
#define FFTOP_OMP_FOR_IF(cond) FFTOP_PRAGMA(omp parallel for schedule(static) if(cond))
#else
#define FFTOP_OMP_FOR_IF(cond)
#endif
