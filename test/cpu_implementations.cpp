#include "utils.h"

#include <fft/cpu/implementations.h>
#include <fft/ref/dft.h>

namespace {

const FFTTest::ImplementationRegistrar kReferenceDft("Reference DFT", dft);
const FFTTest::ImplementationRegistrar kRadix2("Radix-2 FFT", radix_2_fft);
const FFTTest::ImplementationRegistrar kRadix4("Radix-4 FFT", radix_4_fft);
const FFTTest::ImplementationRegistrar kIterative("Iterative In-Place FFT", fft_iterative);
const FFTTest::ImplementationRegistrar kSimdIter("SIMD Iterative In-Place FFT", fft_simd_iterative);
const FFTTest::ImplementationRegistrar kOpenmpIter("OpenMP+SIMD Iterative In-Place FFT", fft_openmp_iterative);
const FFTTest::ImplementationRegistrar kFourStep("Four-Step FFT", fft_four_step);
const FFTTest::ImplementationRegistrar kParallelFourStep(
    "Parallel Four-Step FFT",
    fft_parallel_four_step);
}  // namespace
