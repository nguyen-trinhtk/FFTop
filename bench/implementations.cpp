#include "utils.h"
#include "fftw_wrapper.h"

#include <fft/ref/dft.h>
#include <fft/cpu/radix-2.cpp>
#include <fft/cpu/radix-4.cpp>
#include <fft/cpu/iterative.cpp>
#include <fft/cpu/simd-iter.cpp>
#include <fft/cpu/openmp-iter.cpp>
#include <fft/cpu/four-step.cpp>
#include <fft/cpu/parallel-four-step.cpp>

namespace {
// Reference DFT is O(n^2); skipped in bench at large N.
// const FFTBench::ImplementationRegistrar kReferenceDft("Reference DFT", dft);
const FFTBench::ImplementationRegistrar kRadix2("Radix-2 FFT", radix_2_fft);
const FFTBench::ImplementationRegistrar kRadix4("Radix-4 FFT", radix_4_fft);
const FFTBench::ImplementationRegistrar kIterative("Iterative In-Place FFT", fft_iterative);
const FFTBench::ImplementationRegistrar kSimdIter("SIMD Iterative In-Place FFT", fft_simd_iterative);
const FFTBench::ImplementationRegistrar kOpenmpIter("OpenMP+SIMD Iterative In-Place FFT", fft_openmp_iterative);
const FFTBench::ImplementationRegistrar kFourStep("Four-Step FFT", fft_four_step);
const FFTBench::ImplementationRegistrar kParallelFourStep(
    "Parallel Four-Step FFT",
    fft_parallel_four_step);
const FFTBench::ImplementationRegistrar kFftwCold("FFTW3 (cold)", fftw_fft_cold);
const FFTBench::ImplementationRegistrar kFftwSteady("FFTW3 (steady)", fftw_fft_steady);
}  // namespace
