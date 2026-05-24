#include "utils.h"

#include <fft/ref/dft.h>
#include <fft/cpu/radix-2.cpp>

namespace {

const FFTBench::ImplementationRegistrar kReferenceDft("Reference DFT", dft);
const FFTBench::ImplementationRegistrar kRadix2("Radix-2 FFT", radix_2_fft);
}  // namespace
