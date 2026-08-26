#include "fftop/system/detect.h"

#include <algorithm>
#include <thread>

#if defined(FFTOP_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace FFTop {

bool detect_openmp() {
#if defined(FFTOP_ENABLE_OPENMP)
    return true;
#else
    return false;
#endif
}

std::size_t detect_thread_count() {
    std::size_t n = std::max(1u, std::thread::hardware_concurrency());
#if defined(FFTOP_ENABLE_OPENMP)
    const int omp_n = omp_get_max_threads();
    if (omp_n > 0) n = static_cast<std::size_t>(omp_n);
#endif
    return n;
}

}  // namespace FFTop
