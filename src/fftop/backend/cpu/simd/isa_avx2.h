#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx2,fma")
#endif

#include "fftop/types.h"

#include <immintrin.h>

namespace FFTop::CPU::Avx2 {

using Pack = __m128d;

inline Pack load(const Complex& z) {
    return _mm_loadu_pd(reinterpret_cast<const double*>(&z));
}

inline void store(Complex& z, Pack v) {
    _mm_storeu_pd(reinterpret_cast<double*>(&z), v);
}

inline Pack setc(double re, double im) { return _mm_setr_pd(re, im); }
inline Pack add(Pack a, Pack b)        { return _mm_add_pd(a, b); }
inline Pack sub(Pack a, Pack b)        { return _mm_sub_pd(a, b); }

inline Pack cmul(Pack a, Pack b) {
    const Pack b_flip = _mm_shuffle_pd(b, b, 0x1);
    const Pack a_re   = _mm_movedup_pd(a);
    const Pack a_im   = _mm_permute_pd(a, 0x3);
    const Pack t0     = _mm_mul_pd(a_re, b);
    const Pack t1     = _mm_mul_pd(a_im, b_flip);
    return _mm_fmadd_pd(t1, setc(-1.0, 1.0), t0);
}

inline Pack mul_j(Pack z) {
    return _mm_xor_pd(_mm_shuffle_pd(z, z, 0x1), setc(-0.0, 0.0));
}

inline Pack mul_minus_j(Pack z) {
    return _mm_xor_pd(_mm_shuffle_pd(z, z, 0x1), setc(0.0, -0.0));
}
}  // namespace FFTop::CPU::Avx2
