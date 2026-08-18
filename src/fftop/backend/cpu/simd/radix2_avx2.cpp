#include "fftop/backend/cpu/simd/simd.h"

#include <immintrin.h>
#include <cmath>

namespace FFTop::CPU {
namespace {

inline __m128d load_c(const Complex& z) {
    return _mm_loadu_pd(reinterpret_cast<const double*>(&z));
}

inline void store_c(Complex& z, __m128d v) {
    _mm_storeu_pd(reinterpret_cast<double*>(&z), v);
}

inline __m128d cmul(__m128d a, __m128d b) {
    const __m128d b_flip = _mm_shuffle_pd(b, b, 0x1);
    const __m128d a_re   = _mm_movedup_pd(a);
    const __m128d a_im   = _mm_permute_pd(a, 0x3);
    const __m128d t0     = _mm_mul_pd(a_re, b);
    const __m128d t1     = _mm_mul_pd(a_im, b_flip);
    const __m128d nms    = _mm_setr_pd(-1.0, 1.0);
    return _mm_fmadd_pd(t1, nms, t0);
}

}  // namespace

void radix2_dit_avx2(Complex* data, std::size_t offset, std::size_t stride, double sign) {
    Complex* a = data + offset;
    Complex* b = data + offset + stride;

    for (std::size_t k = 0; k < stride; ++k) {
        const double angle = sign * M_PI * static_cast<double>(k) / static_cast<double>(stride);
        const __m128d w = _mm_setr_pd(std::cos(angle), std::sin(angle));
        const __m128d u = load_c(a[k]);
        const __m128d v = cmul(w, load_c(b[k]));
        store_c(a[k], _mm_add_pd(u, v));
        store_c(b[k], _mm_sub_pd(u, v));
    }
}

}  // namespace FFTop::CPU
