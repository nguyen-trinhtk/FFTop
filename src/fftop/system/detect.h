#pragma once

#include "fftop/system.h"

#include <string>

namespace FFTop {

std::string detect_cpu_name();
Simd        detect_simd();
bool        detect_openmp();
std::size_t detect_thread_count();
bool        detect_nvidia_gpu();

}  // namespace FFTop
