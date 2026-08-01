#include "utils.h"

#include <fft/gpu/implementations.h>

namespace {

const FFTBench::ImplementationRegistrar kGpuNaive("GPU Naive FFT", fft_gpu_naive);
const FFTBench::ImplementationRegistrar kGpuSharedMem(
    "GPU Shared-Mem FFT",
    fft_gpu_shared_mem);
const FFTBench::ImplementationRegistrar kGpuFourStep(
    "GPU Four-Step FFT",
    fft_gpu_four_step);
const FFTBench::ImplementationRegistrar kGpuWarpShuffle(
    "GPU Warp-Shuffle FFT",
    fft_gpu_warp_shuffle);
const FFTBench::ImplementationRegistrar kGpuCufft("cuFFT", fft_gpu_cufft);

}  // namespace
