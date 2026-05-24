#include "utils.h"

#include <fft/ref/dft.h>

namespace {

const FFTBench::ImplementationRegistrar kReferenceDft("Reference DFT", dft);

}  // namespace
