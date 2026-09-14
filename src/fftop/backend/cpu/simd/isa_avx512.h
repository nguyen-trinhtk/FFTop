#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC target("avx512f,fma")
#endif

#include "fftop/types.h"

#include <cstddef>
#include <immintrin.h>

namespace FFTop::CPU::AVX512 {

// Four complexes: [re0, im0, re1, im1, re2, im2, re3, im3]
static constexpr std::size_t width = 4;
using Pack = __m512d;

inline Pack load(const Complex& z) {
    return _mm512_loadu_pd(reinterpret_cast<const double*>(&z));
}

inline void store(Complex& z, Pack v) {
    _mm512_storeu_pd(reinterpret_cast<double*>(&z), v);
}

inline Pack load_pack(const Real* p) { return _mm512_loadu_pd(p); }

inline Pack add(Pack a, Pack b) { return _mm512_add_pd(a, b); }
inline Pack sub(Pack a, Pack b) { return _mm512_sub_pd(a, b); }

inline Pack xor_pd(Pack a, Pack b) {
    return _mm512_castsi512_pd(
        _mm512_xor_si512(_mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}

inline Pack cmul(Pack a, Pack b) {
    const Pack b_flip = _mm512_permute_pd(b, 0x55);
    const Pack a_re   = _mm512_movedup_pd(a);
    const Pack a_im   = _mm512_permute_pd(a, 0xFF);
    const Pack t0     = _mm512_mul_pd(a_re, b);
    const Pack t1     = _mm512_mul_pd(a_im, b_flip);
    const Pack nms    = _mm512_setr_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    return _mm512_fmadd_pd(t1, nms, t0);
}

inline Pack mul_j(Pack z) {
    const Pack signs = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
    return xor_pd(_mm512_permute_pd(z, 0x55), signs);
}

inline Pack mul_minus_j(Pack z) {
    const Pack signs = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return xor_pd(_mm512_permute_pd(z, 0x55), signs);
}

inline Pack conj(Pack z) {
    const Pack signs = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return xor_pd(z, signs);
}

}  // namespace FFTop::CPU::AVX512
