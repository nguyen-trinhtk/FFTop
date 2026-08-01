#include "utils.h"

#include <fft/gpu/implementations.h>

namespace {

const FFTTest::ImplementationRegistrar kGpuNaive(
    "GPU Naive (before)",
    fft_gpu_naive);
const FFTTest::ImplementationRegistrar kGpuOptimized(
    "GPU Optimized (warp/shared locality)",
    fft_gpu_optimized);

}  // namespace
