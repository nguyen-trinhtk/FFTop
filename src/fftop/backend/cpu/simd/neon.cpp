#include "fftop/backend/cpu/simd/kernels.h"

#if defined(FFTOP_NEON)

#include "fftop/backend/cpu/simd/kernels.inl"
#include "fftop/types.h"

#include <arm_neon.h>
#include <cstddef>

namespace FFTop::CPU {
// Since Neon only support 128-bit vector,
// this is technically a scalar-vector for 64-bit complexes
// but use faster ops

// Holds 1 complex per vector
struct NEON {
    static constexpr std::size_t width = 1;
    using Pack                         = float64x2_t;

    static Pack load(const Complex& z) { return vld1q_f64(reinterpret_cast<const double*>(&z)); }
    static void store(Complex& z, Pack v) { vst1q_f64(reinterpret_cast<double*>(&z), v); }

    static Pack add(Pack a, Pack b) { return vaddq_f64(a, b); }
    static Pack sub(Pack a, Pack b) { return vsubq_f64(a, b); }

    static Pack cmul(Pack a, Pack b) {
        const Pack homo   = vmulq_f64(a, conj(b));             // arbr, -aibi
        const Pack hetero = vmulq_f64(vextq_f64(a, a, 1), b);  // [aibr, arbi]
        return vpaddq_f64(homo, hetero);                       // horizontal add
    }

    static Pack mul_j(Pack z) { return vextq_f64(conj(z), conj(z), 1); }
    static Pack mul_minus_j(Pack z) { return conj(vextq_f64(z, z, 1)); }
    static Pack conj(Pack z) {
        // Convert to u64 and flip sign bit of im
        return vreinterpretq_f64_u64(veorq_u64(  // xor to flip sign bit
            vreinterpretq_u64_f64(z), vreinterpretq_u64_f64(Pack{0.0, -0.0})  // signed im
            ));
    }
};

template void radix2<NEON>(Complex*, std::size_t, Direction, const Complex*);
template void radix4<NEON>(Complex*, std::size_t, Direction, const Complex*);

}  // namespace FFTop::CPU

#endif
