#include "fftop/backend/cpu/simd/kernels.h"

#if defined(FFTOP_AVX2)

// Built with -mavx2 -mfma (or /arch:AVX2); see CMakeLists.txt
#include "fftop/backend/cpu/simd/kernels.inl"
#include "fftop/types.h"

#include <cstddef>
#include <immintrin.h>

namespace FFTop::CPU {

// Two interleaved complexes: [re0, im0, re1, im1]
struct AVX2 {
    static constexpr std::size_t width = 2;
    using Pack = __m256d;

    static Pack load(const Complex& z) {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(&z));
    }
    static void store(Complex& z, Pack v) {
        _mm256_storeu_pd(reinterpret_cast<double*>(&z), v);
    }

    static Pack add(Pack a, Pack b) { return _mm256_add_pd(a, b); }
    static Pack sub(Pack a, Pack b) { return _mm256_sub_pd(a, b); }

    static Pack cmul(Pack a, Pack b) {
        const Pack b_flip = _mm256_permute_pd(b, 0x5); // bim, bre
        const Pack a_re   = _mm256_movedup_pd(a);
        const Pack a_im   = _mm256_permute_pd(a, 0xF);
        const Pack t0     = _mm256_mul_pd(a_re, b);
        const Pack t1     = _mm256_mul_pd(a_im, b_flip);
        const Pack nms    = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);
        return _mm256_fmadd_pd(t1, nms, t0);
    }

    static Pack mul_j(Pack z) {
        const Pack signs = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
        return _mm256_xor_pd(_mm256_permute_pd(z, 0x5), signs);
    }
    static Pack mul_minus_j(Pack z) {
        const Pack signs = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        return _mm256_xor_pd(_mm256_permute_pd(z, 0x5), signs);
    }
    static Pack conj(Pack z) {
        const Pack signs = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
        return _mm256_xor_pd(z, signs);
    }
};

template void radix2<AVX2>(Buffer&, std::size_t, std::size_t, Direction, const Complex*);
template void radix4<AVX2>(Buffer&, std::size_t, std::size_t, Direction, const Complex*);

}  // namespace FFTop::CPU

#endif
