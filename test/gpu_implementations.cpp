#include "utils.h"

#include <fft/gpu/implementations.h>

namespace {

const FFTTest::ImplementationRegistrar kGpuNaive("GPU Naive FFT", fft_gpu_naive);
const FFTTest::ImplementationRegistrar kGpuSharedMem(
    "GPU Shared-Mem FFT",
    fft_gpu_shared_mem);
const FFTTest::ImplementationRegistrar kGpuFourStep(
    "GPU Four-Step FFT",
    fft_gpu_four_step);
const FFTTest::ImplementationRegistrar kGpuWarpShuffle(
    "GPU Warp-Shuffle FFT",
    fft_gpu_warp_shuffle);
const FFTTest::ImplementationRegistrar kGpuCufft("cuFFT", fft_gpu_cufft);

}  // namespace
