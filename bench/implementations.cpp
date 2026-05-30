#include "utils.h"

#include <fft/ref/dft.h>
#include <fft/cpu/radix-2.cpp>
#include <fft/cpu/radix-4.cpp>

namespace {

const FFTBench::ImplementationRegistrar kReferenceDft("Reference DFT", dft);
const FFTBench::ImplementationRegistrar kRadix2("Radix-2 FFT", radix_2_fft);
const FFTBench::ImplementationRegistrar kRadix4("Radix-4 FFT", radix_4_fft);
}  // namespace
