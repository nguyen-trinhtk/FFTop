#include "utils.h"

#include <fft/gpu/implementations.h>

namespace {

const FFTTest::ImplementationRegistrar kGpuNaive("GPU Naive FFT", fft_gpu_naive);

}  // namespace
