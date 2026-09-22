#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma")
#endif

#include "fftop/types.h"

#include <cstddef>
#include <immintrin.h>

namespace FFTop::CPU::AVX2 {

// Two complexes: [re0, im0, re1, im1]
static constexpr std::size_t width = 2;
using Pack = __m256d;

inline Pack load(const Complex& z) {
    return _mm256_loadu_pd(reinterpret_cast<const double*>(&z));
}

inline void store(Complex& z, Pack v) {
    _mm256_storeu_pd(reinterpret_cast<double*>(&z), v);
}

inline Pack load_pack(const Real* p) { return _mm256_loadu_pd(p); }

inline Pack add(Pack a, Pack b) { return _mm256_add_pd(a, b); }
inline Pack sub(Pack a, Pack b) { return _mm256_sub_pd(a, b); }

inline Pack cmul(Pack a, Pack b) {
    const Pack b_flip = _mm256_permute_pd(b, 0x5);
    const Pack a_re   = _mm256_movedup_pd(a);
    const Pack a_im   = _mm256_permute_pd(a, 0xF);
    const Pack t0     = _mm256_mul_pd(a_re, b);
    const Pack t1     = _mm256_mul_pd(a_im, b_flip);
    const Pack nms    = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
    return _mm256_fmadd_pd(t1, nms, t0);
}

inline Pack mul_j(Pack z) {
    const Pack signs = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    return _mm256_xor_pd(_mm256_permute_pd(z, 0x5), signs);
}

inline Pack mul_minus_j(Pack z) {
    const Pack signs = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_pd(_mm256_permute_pd(z, 0x5), signs);
}

inline Pack conj(Pack z) {
    const Pack signs = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_pd(z, signs);
}

}  // namespace FFTop::CPU::AVX2
