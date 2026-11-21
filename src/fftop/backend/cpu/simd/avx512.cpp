#include "fftop/backend/cpu/simd/kernels.h"

#if defined(FFTOP_AVX512)

// Built with -mavx512f -mfma (or /arch:AVX512); see CMakeLists.txt
#include "fftop/backend/cpu/simd/kernels.inl"
#include "fftop/types.h"

#include <cstddef>
#include <immintrin.h>

namespace FFTop::CPU {

// AVX512 can hold four interleaved complexes
struct AVX512 {
    static constexpr std::size_t width = 4;
    using Pack = __m512d;

    static Pack load(const Complex& z) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(&z));
    }
    static void store(Complex& z, Pack v) {
        _mm512_storeu_pd(reinterpret_cast<double*>(&z), v);
    }

    static Pack add(Pack a, Pack b) { return _mm512_add_pd(a, b); }
    static Pack sub(Pack a, Pack b) { return _mm512_sub_pd(a, b); }

    // Floating-point XOR on AVX512F
    // note that AVX0512DQ and later supports xor_pd
    static Pack xor_pd(Pack a, Pack b) {
        // Cast floating point vector as int, and perform bitwise XOR
        return _mm512_castsi512_pd(
            _mm512_xor_si512(
                _mm512_castpd_si512(a), 
                _mm512_castpd_si512(b)
            )
        );
    }

    static Pack cmul(Pack a, Pack b) {
        const Pack b_flip = _mm512_permute_pd(b, 0x55); // swap elements in 128bit lane
        const Pack a_re   = _mm512_movedup_pd(a); // = permute(a, 0x00)
        const Pack a_im   = _mm512_permute_pd(a, 0xFF);
        const Pack t0     = _mm512_mul_pd(a_re, b);
        const Pack t1     = _mm512_mul_pd(a_im, b_flip);
        const Pack nms    = _mm512_setr_pd(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        return _mm512_fmadd_pd(t1, nms, t0); // arbr-aibi, arbi+aibr
    }

    static Pack mul_j(Pack z) {
        const Pack signs = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        return xor_pd(_mm512_permute_pd(z, 0x55), signs); // flip & negate re
    }
    static Pack mul_minus_j(Pack z) {
        const Pack signs = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        return xor_pd(_mm512_permute_pd(z, 0x55), signs); // flip & negate im
    }
    static Pack conj(Pack z) {
        const Pack signs = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
        return xor_pd(z, signs); // negate im
    }
};

template void radix2<AVX512>(Buffer&, std::size_t, std::size_t, Direction, const Complex*);
template void radix4<AVX512>(Buffer&, std::size_t, std::size_t, Direction, const Complex*);
}  // namespace FFTop::CPU

#endif
