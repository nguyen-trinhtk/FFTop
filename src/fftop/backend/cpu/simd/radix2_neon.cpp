#include "fftop/backend/cpu/simd/simd.h"

#include <arm_neon.h>
#include <cmath>

namespace FFTop::CPU {
namespace {

inline float64x2_t load_c(const Complex& z) {
    return vld1q_f64(reinterpret_cast<const double*>(&z));
}

inline void store_c(Complex& z, float64x2_t v) {
    vst1q_f64(reinterpret_cast<double*>(&z), v);
}

inline float64x2_t cmul(float64x2_t a, float64x2_t b) {
    const float64x2_t b_flip = vextq_f64(b, b, 1);
    const float64x2_t a_re   = vdupq_laneq_f64(a, 0);
    const float64x2_t a_im   = vdupq_laneq_f64(a, 1);
    const float64x2_t t0     = vmulq_f64(a_re, b);
    const float64x2_t t1     = vmulq_f64(a_im, b_flip);
    const float64x2_t nms    = {-1.0, 1.0};
    return vfmaq_f64(t0, t1, nms);
}

}  // namespace

void radix2_dit_neon(Complex* data, std::size_t offset, std::size_t stride, double sign) {
    Complex* a = data + offset;
    Complex* b = data + offset + stride;

    for (std::size_t k = 0; k < stride; ++k) {
        const double angle = sign * M_PI * static_cast<double>(k) / static_cast<double>(stride);
        const float64x2_t w = vsetq_lane_f64(std::sin(angle), vdupq_n_f64(std::cos(angle)), 1);
        const float64x2_t u = load_c(a[k]);
        const float64x2_t v = cmul(w, load_c(b[k]));
        store_c(a[k], vaddq_f64(u, v));
        store_c(b[k], vsubq_f64(u, v));
    }
}

}  // namespace FFTop::CPU
