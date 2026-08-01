#include "utils.h"

#include <fft/gpu/implementations.h>

namespace {

const FFTBench::ImplementationRegistrar kGpuNaive(
    "GPU Naive (before)",
    fft_gpu_naive);
const FFTBench::ImplementationRegistrar kGpuOptimized(
    "GPU Optimized (warp/shared locality)",
    fft_gpu_optimized);

}  // namespace
