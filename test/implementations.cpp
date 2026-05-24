#include "utils.h"
#include <fft/ref/dft.h>
#include <fft/cpu/radix-2.cpp>

namespace {

const FFTTest::ImplementationRegistrar kReferenceDft("Reference DFT", dft);
const FFTTest::ImplementationRegistrar kRadix2("Radix-2 FFT", radix_2_fft);

}  // namespace
