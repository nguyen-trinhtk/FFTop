#pragma once

#include "fftop/types.h"

#include <arm_neon.h>

namespace FFTop::CPU::ISA {

using Pack = float64x2_t;

inline Pack load(const Complex& z) {
    return vld1q_f64(reinterpret_cast<const double*>(&z));
}

inline void store(Complex& z, Pack v) {
    vst1q_f64(reinterpret_cast<double*>(&z), v);
}

inline Pack setc(double re, double im) {
    return vsetq_lane_f64(im, vdupq_n_f64(re), 1);
}

inline Pack add(Pack a, Pack b) { return vaddq_f64(a, b); }
inline Pack sub(Pack a, Pack b) { return vsubq_f64(a, b); }

inline Pack cmul(Pack a, Pack b) {
    const Pack b_flip = vextq_f64(b, b, 1);
    const Pack a_re   = vdupq_laneq_f64(a, 0);
    const Pack a_im   = vdupq_laneq_f64(a, 1);
    const Pack t0     = vmulq_f64(a_re, b);
    const Pack t1     = vmulq_f64(a_im, b_flip);
    const Pack nms    = {-1.0, 1.0};
    return vfmaq_f64(t0, t1, nms);
}

inline Pack mul_j(Pack z) {
    const Pack swapped = vextq_f64(z, z, 1);
    const Pack nps     = {-1.0, 1.0};
    return vmulq_f64(swapped, nps);
}

inline Pack mul_minus_j(Pack z) {
    const Pack swapped = vextq_f64(z, z, 1);
    const Pack pns     = {1.0, -1.0};
    return vmulq_f64(swapped, pns);
}
}  // namespace FFTop::CPU::ISA
