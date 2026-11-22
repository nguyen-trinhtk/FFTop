#pragma once

#include <cstddef>

namespace FFTop::CPU {
using omp_index = std::ptrdiff_t;  // (Signed) loop index for FFTOP_OMP_FOR_IF loops

// Should we parallelize a work or no?
inline bool use_parallel(int threads, std::size_t n) {
    return threads > 1 && n > 1;  // well planner have done most of the decisions
}
}  // namespace FFTop::CPU

// Put this directive before a for loop that can possibly
// be parallelized when OpenMP is enabled and cond is true;
// nthreads applies to this loop only
#define FFTOP_PRAGMA(x) _Pragma(#x)
#ifdef FFTOP_ENABLE_OPENMP
#define FFTOP_OMP_FOR_IF(cond, nthreads) \
    FFTOP_PRAGMA(omp parallel for schedule(static) if(cond) num_threads(nthreads))
#else
#define FFTOP_OMP_FOR_IF(cond, nthreads) ((void)(cond), (void)(nthreads));
#endif
