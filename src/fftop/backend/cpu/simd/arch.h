#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFTOP_HAS_AVX2_KERNEL 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define FFTOP_HAS_NEON_KERNEL 1
#endif
