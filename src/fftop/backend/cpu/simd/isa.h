#pragma once

// ISA-specific headers
#if defined(FFTOP_ENABLE_AVX2)
#include "fftop/backend/cpu/simd/isa_avx2.h"
#elif defined(FFTOP_ENABLE_NEON)
#include "fftop/backend/cpu/simd/isa_neon.h"
#else
#include "fftop/backend/cpu/simd/isa_scalar.h"
#endif
