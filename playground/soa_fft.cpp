// Full iterative DIT FFT: interleaved AoS vs split SoA, scalar vs NEON.
// Matches FFTop: one length-N twiddle table, unnormalized inverse, vectorize k
// inside each butterfly. SoA is a full layout change (re[], im[], Wr[], Wi[]).
//
//   clang++ -O3 -std=c++17 playground/soa_fft.cpp -o /tmp/soa_fft && /tmp/soa_fft

#include <arm_neon.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#if defined(__clang__)
#define NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define NOVEC
#endif

using Complex = std::complex<double>;

enum class Dir { Fwd, Inv };

inline std::size_t digit_reverse(std::size_t index, std::size_t n, std::size_t radix) {
    const std::size_t digit_mask = radix - 1;
    unsigned          bits       = 0;
    for (std::size_t r = radix; r > 1; r >>= 1) ++bits;
    std::size_t reversed = 0;
    for (std::size_t rest = n; rest > 1; rest >>= bits) {
        reversed = (reversed << bits) | (index & digit_mask);
        index >>= bits;
    }
    return reversed;
}

void bitrev_aos(Complex* x, std::size_t n, std::size_t radix) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = digit_reverse(i, n, radix);
        if (i < j) std::swap(x[i], x[j]);
    }
}

void bitrev_soa(double* re, double* im, std::size_t n, std::size_t radix) {
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = digit_reverse(i, n, radix);
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
}

void make_twiddles_aos(Complex* W, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        const double a = -6.28318530717958647692 * double(k) / double(n);
        W[k]           = {std::cos(a), std::sin(a)};
    }
}

void make_twiddles_soa(double* Wr, double* Wi, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        const double a = -6.28318530717958647692 * double(k) / double(n);
        Wr[k]          = std::cos(a);
        Wi[k]          = std::sin(a);
    }
}

void split(const Complex* x, double* re, double* im, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        re[i] = x[i].real();
        im[i] = x[i].imag();
    }
}

void join(const double* re, const double* im, Complex* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) x[i] = {re[i], im[i]};
}

template <class T>
inline void keep(T& x) {
    asm volatile("" : "+m"(x)::"memory");
}

inline void cmul(double wr, double wi, double br, double bi, double& tr, double& ti) {
    tr = wr * br - wi * bi;
    ti = wr * bi + wi * br;
}

inline void tw(const Complex* W, std::size_t k, std::size_t step, Dir dir, double& wr, double& wi) {
    wr = W[k * step].real();
    wi = W[k * step].imag();
    if (dir == Dir::Inv) wi = -wi;
}

inline void tw(const double* Wr, const double* Wi, std::size_t k, std::size_t step, Dir dir,
               double& wr, double& wi) {
    wr = Wr[k * step];
    wi = Wi[k * step];
    if (dir == Dir::Inv) wi = -wi;
}

inline float64x2_t load2(const double* p, std::size_t k, std::size_t step) {
    if (step == 1) return vld1q_f64(p + k);
    return float64x2_t{p[k * step], p[(k + 1) * step]};
}

inline float64x2_t maybe_neg(float64x2_t v, Dir dir) {
    if (dir == Dir::Inv) return vnegq_f64(v);
    return v;
}

inline float64x2_t cmul_aos(float64x2_t w, float64x2_t b) {
    const float64x2_t b_flip = vextq_f64(b, b, 1);
    const float64x2_t wr     = vdupq_laneq_f64(w, 0);
    const float64x2_t wi     = vdupq_laneq_f64(w, 1);
    const float64x2_t nms    = {-1.0, 1.0};
    return vfmaq_f64(vmulq_f64(wr, b), vmulq_f64(wi, b_flip), nms);
}

inline float64x2_t conj_aos(float64x2_t z) {
    const uint64x2_t s = vreinterpretq_u64_f64(float64x2_t{0.0, -0.0});
    return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(z), s));
}

// --- radix-2 inner (one group) ---

template <bool Autovec>
void r2_aos_group(Complex* x, std::size_t off, std::size_t stride, Dir dir, const Complex* W,
                  std::size_t n) {
    Complex*          x0   = x + off;
    Complex*          x1   = x0 + stride;
    const std::size_t step = n / (2 * stride);
    if constexpr (!Autovec) {
        NOVEC
        for (std::size_t k = 0; k < stride; ++k) {
            double wr, wi, tr, ti;
            tw(W, k, step, dir, wr, wi);
            cmul(wr, wi, x1[k].real(), x1[k].imag(), tr, ti);
            const double ar = x0[k].real(), ai = x0[k].imag();
            x0[k] = {ar + tr, ai + ti};
            x1[k] = {ar - tr, ai - ti};
        }
    } else {
        for (std::size_t k = 0; k < stride; ++k) {
            double wr, wi, tr, ti;
            tw(W, k, step, dir, wr, wi);
            cmul(wr, wi, x1[k].real(), x1[k].imag(), tr, ti);
            const double ar = x0[k].real(), ai = x0[k].imag();
            x0[k] = {ar + tr, ai + ti};
            x1[k] = {ar - tr, ai - ti};
        }
    }
}

void r2_aos_neon_group(Complex* x, std::size_t off, std::size_t stride, Dir dir, const Complex* W,
                       std::size_t n) {
    Complex*          x0   = x + off;
    Complex*          x1   = x0 + stride;
    const std::size_t step = n / (2 * stride);
    for (std::size_t k = 0; k < stride; ++k) {
        float64x2_t w = vld1q_f64(reinterpret_cast<const double*>(&W[k * step]));
        if (dir == Dir::Inv) w = conj_aos(w);
        const float64x2_t a0 = vld1q_f64(reinterpret_cast<double*>(&x0[k]));
        const float64x2_t t  = cmul_aos(w, vld1q_f64(reinterpret_cast<double*>(&x1[k])));
        vst1q_f64(reinterpret_cast<double*>(&x0[k]), vaddq_f64(a0, t));
        vst1q_f64(reinterpret_cast<double*>(&x1[k]), vsubq_f64(a0, t));
    }
}

template <bool Autovec>
void r2_soa_group(double* re, double* im, std::size_t off, std::size_t stride, Dir dir,
                  const double* Wr, const double* Wi, std::size_t n) {
    double*           r0   = re + off;
    double*           i0   = im + off;
    double*           r1   = r0 + stride;
    double*           i1   = i0 + stride;
    const std::size_t step = n / (2 * stride);
    if constexpr (!Autovec) {
        NOVEC
        for (std::size_t k = 0; k < stride; ++k) {
            double wr, wi, tr, ti;
            tw(Wr, Wi, k, step, dir, wr, wi);
            cmul(wr, wi, r1[k], i1[k], tr, ti);
            const double ar = r0[k], ai = i0[k];
            r0[k] = ar + tr;
            i0[k] = ai + ti;
            r1[k] = ar - tr;
            i1[k] = ai - ti;
        }
    } else {
        for (std::size_t k = 0; k < stride; ++k) {
            double wr, wi, tr, ti;
            tw(Wr, Wi, k, step, dir, wr, wi);
            cmul(wr, wi, r1[k], i1[k], tr, ti);
            const double ar = r0[k], ai = i0[k];
            r0[k] = ar + tr;
            i0[k] = ai + ti;
            r1[k] = ar - tr;
            i1[k] = ai - ti;
        }
    }
}

void r2_soa_neon_group(double* re, double* im, std::size_t off, std::size_t stride, Dir dir,
                       const double* Wr, const double* Wi, std::size_t n) {
    double*           r0   = re + off;
    double*           i0   = im + off;
    double*           r1   = r0 + stride;
    double*           i1   = i0 + stride;
    const std::size_t step = n / (2 * stride);
    std::size_t       k    = 0;
    for (; k + 2 <= stride; k += 2) {
        const float64x2_t wr = maybe_neg(load2(Wr, k, step), Dir::Fwd);
        const float64x2_t wi = maybe_neg(load2(Wi, k, step), dir);
        const float64x2_t ar = vld1q_f64(r0 + k);
        const float64x2_t ai = vld1q_f64(i0 + k);
        const float64x2_t br = vld1q_f64(r1 + k);
        const float64x2_t bi = vld1q_f64(i1 + k);
        const float64x2_t tr = vfmsq_f64(vmulq_f64(wr, br), wi, bi);
        const float64x2_t ti = vfmaq_f64(vmulq_f64(wr, bi), wi, br);
        vst1q_f64(r0 + k, vaddq_f64(ar, tr));
        vst1q_f64(i0 + k, vaddq_f64(ai, ti));
        vst1q_f64(r1 + k, vsubq_f64(ar, tr));
        vst1q_f64(i1 + k, vsubq_f64(ai, ti));
    }
    for (; k < stride; ++k) {
        double wr, wi, tr, ti;
        tw(Wr, Wi, k, step, dir, wr, wi);
        cmul(wr, wi, r1[k], i1[k], tr, ti);
        const double ar = r0[k], ai = i0[k];
        r0[k] = ar + tr;
        i0[k] = ai + ti;
        r1[k] = ar - tr;
        i1[k] = ai - ti;
    }
}

// --- radix-4 inner ---

template <bool Autovec>
void r4_aos_group(Complex* x, std::size_t off, std::size_t stride, Dir dir, const Complex* W,
                  std::size_t n) {
    Complex*          x0      = x + off;
    Complex*          x1      = x0 + stride;
    Complex*          x2      = x1 + stride;
    Complex*          x3      = x2 + stride;
    const std::size_t step    = n / (4 * stride);
    const bool        forward = dir == Dir::Fwd;
    auto              body    = [&](std::size_t k) {
        double wr, wi, t1r, t1i, t2r, t2i, t3r, t3i;
        tw(W, k, step, dir, wr, wi);
        cmul(wr, wi, x1[k].real(), x1[k].imag(), t1r, t1i);
        tw(W, 2 * k, step, dir, wr, wi);
        cmul(wr, wi, x2[k].real(), x2[k].imag(), t2r, t2i);
        tw(W, 3 * k, step, dir, wr, wi);
        cmul(wr, wi, x3[k].real(), x3[k].imag(), t3r, t3i);
        const double esr = x0[k].real() + t2r, esi = x0[k].imag() + t2i;
        const double edr = x0[k].real() - t2r, edi = x0[k].imag() - t2i;
        const double osr = t1r + t3r, osi = t1i + t3i;
        const double odr = t1r - t3r, odi = t1i - t3i;
        const double otr = forward ? odi : -odi;
        const double oti = forward ? -odr : odr;
        x0[k]            = {esr + osr, esi + osi};
        x1[k]            = {edr + otr, edi + oti};
        x2[k]            = {esr - osr, esi - osi};
        x3[k]            = {edr - otr, edi - oti};
    };
    if constexpr (!Autovec) {
        NOVEC
        for (std::size_t k = 0; k < stride; ++k) body(k);
    } else {
        for (std::size_t k = 0; k < stride; ++k) body(k);
    }
}

template <bool Autovec>
void r4_soa_group(double* re, double* im, std::size_t off, std::size_t stride, Dir dir,
                  const double* Wr, const double* Wi, std::size_t n) {
    double*           r0      = re + off;
    double*           i0      = im + off;
    double*           r1      = r0 + stride;
    double*           i1      = i0 + stride;
    double*           r2      = r1 + stride;
    double*           i2      = i1 + stride;
    double*           r3      = r2 + stride;
    double*           i3      = i2 + stride;
    const std::size_t step    = n / (4 * stride);
    const bool        forward = dir == Dir::Fwd;
    auto              body    = [&](std::size_t k) {
        double wr, wi, t1r, t1i, t2r, t2i, t3r, t3i;
        tw(Wr, Wi, k, step, dir, wr, wi);
        cmul(wr, wi, r1[k], i1[k], t1r, t1i);
        tw(Wr, Wi, 2 * k, step, dir, wr, wi);
        cmul(wr, wi, r2[k], i2[k], t2r, t2i);
        tw(Wr, Wi, 3 * k, step, dir, wr, wi);
        cmul(wr, wi, r3[k], i3[k], t3r, t3i);
        const double esr = r0[k] + t2r, esi = i0[k] + t2i;
        const double edr = r0[k] - t2r, edi = i0[k] - t2i;
        const double osr = t1r + t3r, osi = t1i + t3i;
        const double odr = t1r - t3r, odi = t1i - t3i;
        const double otr = forward ? odi : -odi;
        const double oti = forward ? -odr : odr;
        r0[k]            = esr + osr;
        i0[k]            = esi + osi;
        r1[k]            = edr + otr;
        i1[k]            = edi + oti;
        r2[k]            = esr - osr;
        i2[k]            = esi - osi;
        r3[k]            = edr - otr;
        i3[k]            = edi - oti;
    };
    if constexpr (!Autovec) {
        NOVEC
        for (std::size_t k = 0; k < stride; ++k) body(k);
    } else {
        for (std::size_t k = 0; k < stride; ++k) body(k);
    }
}

inline float64x2_t load2_scaled(const double* p, std::size_t k, std::size_t mul, std::size_t step) {
    // W[mul * (k + i) * step] for i=0,1
    const std::size_t s = mul * step;
    if (s == 1) return vld1q_f64(p + mul * k);
    return float64x2_t{p[mul * k * step], p[mul * (k + 1) * step]};
}

void r4_aos_neon_group(Complex* x, std::size_t off, std::size_t stride, Dir dir, const Complex* W,
                       std::size_t n) {
    Complex*          x0      = x + off;
    Complex*          x1      = x0 + stride;
    Complex*          x2      = x1 + stride;
    Complex*          x3      = x2 + stride;
    const std::size_t step    = n / (4 * stride);
    const bool        forward = dir == Dir::Fwd;
    for (std::size_t k = 0; k < stride; ++k) {
        auto lw = [&](std::size_t m) {
            float64x2_t w = vld1q_f64(reinterpret_cast<const double*>(&W[m * k * step]));
            return dir == Dir::Inv ? conj_aos(w) : w;
        };
        const float64x2_t a0 = vld1q_f64(reinterpret_cast<double*>(&x0[k]));
        const float64x2_t a1 = cmul_aos(lw(1), vld1q_f64(reinterpret_cast<double*>(&x1[k])));
        const float64x2_t a2 = cmul_aos(lw(2), vld1q_f64(reinterpret_cast<double*>(&x2[k])));
        const float64x2_t a3 = cmul_aos(lw(3), vld1q_f64(reinterpret_cast<double*>(&x3[k])));
        const float64x2_t es = vaddq_f64(a0, a2);
        const float64x2_t ed = vsubq_f64(a0, a2);
        const float64x2_t os = vaddq_f64(a1, a3);
        const float64x2_t od = vsubq_f64(a1, a3);
        const float64x2_t sw = vextq_f64(od, od, 1);
        const float64x2_t ot =
            vmulq_f64(sw, forward ? float64x2_t{1.0, -1.0} : float64x2_t{-1.0, 1.0});
        vst1q_f64(reinterpret_cast<double*>(&x0[k]), vaddq_f64(es, os));
        vst1q_f64(reinterpret_cast<double*>(&x1[k]), vaddq_f64(ed, ot));
        vst1q_f64(reinterpret_cast<double*>(&x2[k]), vsubq_f64(es, os));
        vst1q_f64(reinterpret_cast<double*>(&x3[k]), vsubq_f64(ed, ot));
    }
}

void r4_soa_neon_group(double* re, double* im, std::size_t off, std::size_t stride, Dir dir,
                       const double* Wr, const double* Wi, std::size_t n) {
    double*           r0      = re + off;
    double*           i0      = im + off;
    double*           r1      = r0 + stride;
    double*           i1      = i0 + stride;
    double*           r2      = r1 + stride;
    double*           i2      = i1 + stride;
    double*           r3      = r2 + stride;
    double*           i3      = i2 + stride;
    const std::size_t step    = n / (4 * stride);
    const bool        forward = dir == Dir::Fwd;
    std::size_t       k       = 0;
    for (; k + 2 <= stride; k += 2) {
        auto wmul = [&](std::size_t mul) {
            const float64x2_t wr  = load2_scaled(Wr, k, mul, step);
            const float64x2_t wi  = maybe_neg(load2_scaled(Wi, k, mul, step), dir);
            const double*     br  = (mul == 1 ? r1 : mul == 2 ? r2 : r3);
            const double*     bi  = (mul == 1 ? i1 : mul == 2 ? i2 : i3);
            const float64x2_t b_r = vld1q_f64(br + k);
            const float64x2_t b_i = vld1q_f64(bi + k);
            return std::pair<float64x2_t, float64x2_t>{
                vfmsq_f64(vmulq_f64(wr, b_r), wi, b_i),
                vfmaq_f64(vmulq_f64(wr, b_i), wi, b_r),
            };
        };
        const float64x2_t ar  = vld1q_f64(r0 + k);
        const float64x2_t ai  = vld1q_f64(i0 + k);
        const auto        a1  = wmul(1);
        const auto        a2  = wmul(2);
        const auto        a3  = wmul(3);
        const float64x2_t esr = vaddq_f64(ar, a2.first);
        const float64x2_t esi = vaddq_f64(ai, a2.second);
        const float64x2_t edr = vsubq_f64(ar, a2.first);
        const float64x2_t edi = vsubq_f64(ai, a2.second);
        const float64x2_t osr = vaddq_f64(a1.first, a3.first);
        const float64x2_t osi = vaddq_f64(a1.second, a3.second);
        const float64x2_t odr = vsubq_f64(a1.first, a3.first);
        const float64x2_t odi = vsubq_f64(a1.second, a3.second);
        const float64x2_t otr = forward ? odi : vnegq_f64(odi);
        const float64x2_t oti = forward ? vnegq_f64(odr) : odr;
        vst1q_f64(r0 + k, vaddq_f64(esr, osr));
        vst1q_f64(i0 + k, vaddq_f64(esi, osi));
        vst1q_f64(r1 + k, vaddq_f64(edr, otr));
        vst1q_f64(i1 + k, vaddq_f64(edi, oti));
        vst1q_f64(r2 + k, vsubq_f64(esr, osr));
        vst1q_f64(i2 + k, vsubq_f64(esi, osi));
        vst1q_f64(r3 + k, vsubq_f64(edr, otr));
        vst1q_f64(i3 + k, vsubq_f64(edi, oti));
    }
    for (; k < stride; ++k) {
        double wr, wi, t1r, t1i, t2r, t2i, t3r, t3i;
        tw(Wr, Wi, k, step, dir, wr, wi);
        cmul(wr, wi, r1[k], i1[k], t1r, t1i);
        tw(Wr, Wi, 2 * k, step, dir, wr, wi);
        cmul(wr, wi, r2[k], i2[k], t2r, t2i);
        tw(Wr, Wi, 3 * k, step, dir, wr, wi);
        cmul(wr, wi, r3[k], i3[k], t3r, t3i);
        const double esr = r0[k] + t2r, esi = i0[k] + t2i;
        const double edr = r0[k] - t2r, edi = i0[k] - t2i;
        const double osr = t1r + t3r, osi = t1i + t3i;
        const double odr = t1r - t3r, odi = t1i - t3i;
        const double otr = forward ? odi : -odi;
        const double oti = forward ? -odr : odr;
        r0[k]            = esr + osr;
        i0[k]            = esi + osi;
        r1[k]            = edr + otr;
        i1[k]            = edi + oti;
        r2[k]            = esr - osr;
        i2[k]            = esi - osi;
        r3[k]            = edr - otr;
        i3[k]            = edi - oti;
    }
}

// --- stage loops ---

template <class Group>
void run_stages(std::size_t n, std::size_t radix, Group&& group) {
    for (std::size_t stride = 1; stride < n; stride *= radix) {
        const std::size_t gs = stride * radix;
        for (std::size_t g = 0; g < n / gs; ++g) group(g * gs, stride);
    }
}

void fft_aos_r2_scalar(Complex* x, std::size_t n, Dir dir, const Complex* W) {
    bitrev_aos(x, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_aos_group<false>(x, off, st, dir, W, n);
    });
}
void fft_aos_r2_autovec(Complex* x, std::size_t n, Dir dir, const Complex* W) {
    bitrev_aos(x, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_aos_group<true>(x, off, st, dir, W, n);
    });
}
void fft_aos_r2_neon(Complex* x, std::size_t n, Dir dir, const Complex* W) {
    bitrev_aos(x, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_aos_neon_group(x, off, st, dir, W, n);
    });
}
void fft_soa_r2_scalar(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                       const double* Wi) {
    bitrev_soa(re, im, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_soa_group<false>(re, im, off, st, dir, Wr, Wi, n);
    });
}
void fft_soa_r2_autovec(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                        const double* Wi) {
    bitrev_soa(re, im, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_soa_group<true>(re, im, off, st, dir, Wr, Wi, n);
    });
}
void fft_soa_r2_neon(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                     const double* Wi) {
    bitrev_soa(re, im, n, 2);
    run_stages(n, 2, [&](std::size_t off, std::size_t st) {
        r2_soa_neon_group(re, im, off, st, dir, Wr, Wi, n);
    });
}

void fft_aos_r4_scalar(Complex* x, std::size_t n, Dir dir, const Complex* W) {
    bitrev_aos(x, n, 4);
    run_stages(n, 4, [&](std::size_t off, std::size_t st) {
        r4_aos_group<false>(x, off, st, dir, W, n);
    });
}
void fft_aos_r4_neon(Complex* x, std::size_t n, Dir dir, const Complex* W) {
    bitrev_aos(x, n, 4);
    run_stages(n, 4, [&](std::size_t off, std::size_t st) {
        r4_aos_neon_group(x, off, st, dir, W, n);
    });
}
void fft_soa_r4_scalar(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                       const double* Wi) {
    bitrev_soa(re, im, n, 4);
    run_stages(n, 4, [&](std::size_t off, std::size_t st) {
        r4_soa_group<false>(re, im, off, st, dir, Wr, Wi, n);
    });
}
void fft_soa_r4_autovec(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                        const double* Wi) {
    bitrev_soa(re, im, n, 4);
    run_stages(n, 4, [&](std::size_t off, std::size_t st) {
        r4_soa_group<true>(re, im, off, st, dir, Wr, Wi, n);
    });
}
void fft_soa_r4_neon(double* re, double* im, std::size_t n, Dir dir, const double* Wr,
                     const double* Wi) {
    bitrev_soa(re, im, n, 4);
    run_stages(n, 4, [&](std::size_t off, std::size_t st) {
        r4_soa_neon_group(re, im, off, st, dir, Wr, Wi, n);
    });
}

void mixed(Complex* x, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        x[i] = {std::cos(double(i) * 0.7), std::sin(double(i) * 1.3) + 0.25};
}

void dft(const Complex* x, Complex* X, std::size_t n, Dir dir) {
    const double sign = dir == Dir::Fwd ? -1.0 : 1.0;
    for (std::size_t k = 0; k < n; ++k) {
        Complex s = {0, 0};
        for (std::size_t t = 0; t < n; ++t) {
            const double a = sign * 6.28318530717958647692 * double(t) * double(k) / double(n);
            s += x[t] * Complex{std::cos(a), std::sin(a)};
        }
        X[k] = s;
    }
}

double rel_err(const Complex* a, const Complex* b, std::size_t n) {
    double err = 0, scale = 0;
    for (std::size_t i = 0; i < n; ++i) {
        err   = std::max(err, std::abs(a[i] - b[i]));
        scale = std::max(scale, std::abs(b[i]));
    }
    return scale > 0 ? err / scale : err;
}

using Clock = std::chrono::steady_clock;

template <class Fn>
double ns_per_sample(Fn&& fn, std::size_t n) {
    int inner = 1;
    fn();
    for (;;) {
        const auto t0 = Clock::now();
        for (int i = 0; i < inner; ++i) fn();
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms >= 25.0 || inner >= 1 << 18) {
            double best = ms;
            for (int r = 0; r < 3; ++r) {
                const auto a = Clock::now();
                for (int i = 0; i < inner; ++i) fn();
                const auto b = Clock::now();
                const double m = std::chrono::duration<double, std::milli>(b - a).count();
                if (m < best) best = m;
            }
            return best * 1e6 / (double(inner) * double(n));
        }
        inner *= 2;
    }
}

int check_one(const char* name, const std::vector<Complex>& got, const std::vector<Complex>& want) {
    const double e = rel_err(got.data(), want.data(), got.size());
    const bool   ok = e < 1e-11;
    std::printf("  %-22s  relerr=%.3e  %s\n", name, e, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    int fails = 0;
    std::printf("correctness vs DFT (N=64) and vs AoS scalar (N=256)\n");

    auto verify = [&](std::size_t n, Dir dir) {
        std::vector<Complex> src(n), want(n), aos(n), out(n);
        std::vector<double>  re(n), im(n), Wr(n), Wi(n);
        std::vector<Complex> W(n);
        mixed(src.data(), n);
        make_twiddles_aos(W.data(), n);
        make_twiddles_soa(Wr.data(), Wi.data(), n);
        dft(src.data(), want.data(), n, dir);
        const char* dname = dir == Dir::Fwd ? "fwd" : "inv";

        auto run_aos = [&](auto fft, const char* name) {
            aos = src;
            fft(aos.data(), n, dir, W.data());
            fails += check_one((std::string(name) + " " + dname).c_str(), aos, want);
        };
        auto run_soa = [&](auto fft, const char* name) {
            split(src.data(), re.data(), im.data(), n);
            fft(re.data(), im.data(), n, dir, Wr.data(), Wi.data());
            join(re.data(), im.data(), out.data(), n);
            fails += check_one((std::string(name) + " " + dname).c_str(), out, want);
        };

        if (n == 64 || (n == 256 && dir == Dir::Fwd)) {
            std::printf("N=%zu %s\n", n, dname);
            run_aos(fft_aos_r2_scalar, "AoS r2 scalar");
            run_aos(fft_aos_r2_autovec, "AoS r2 autovec");
            run_aos(fft_aos_r2_neon, "AoS r2 NEON");
            run_soa(fft_soa_r2_scalar, "SoA r2 scalar");
            run_soa(fft_soa_r2_autovec, "SoA r2 autovec");
            run_soa(fft_soa_r2_neon, "SoA r2 NEON");
            if ((n & (n - 1)) == 0 && n % 4 == 0) {
                run_aos(fft_aos_r4_scalar, "AoS r4 scalar");
                run_aos(fft_aos_r4_neon, "AoS r4 NEON");
                run_soa(fft_soa_r4_scalar, "SoA r4 scalar");
                run_soa(fft_soa_r4_autovec, "SoA r4 autovec");
                run_soa(fft_soa_r4_neon, "SoA r4 NEON");
            }
        }
    };
    verify(64, Dir::Fwd);
    verify(64, Dir::Inv);
    verify(256, Dir::Fwd);

    if (fails) {
        std::printf("\n%d checks failed; skip bench\n", fails);
        return 1;
    }

    std::printf("\nhost: Apple Silicon NEON  (ns/sample, full iterative FFT, twiddles cached)\n");
    std::printf("%8s %10s %10s %10s %10s %10s %10s %12s\n", "N", "AoS sc", "SoA sc", "AoS auto",
                "SoA auto", "AoS NEON", "SoA NEON", "cvt+SoA N");

    const std::size_t sizes[] = {256, 1024, 4096, 16384, 65536, 262144};

    struct Row {
        std::size_t n;
        double      aos_s, soa_s, aos_av, soa_av, aos_n, soa_n, cvt;
        double      r4_aos_s, r4_soa_s, r4_aos_n, r4_soa_n, r4_cvt;
    };
    std::vector<Row> rows;

    for (std::size_t n : sizes) {
        std::vector<Complex> src(n), aos(n), W(n);
        std::vector<double>  re(n), im(n), Wr(n), Wi(n);
        mixed(src.data(), n);
        make_twiddles_aos(W.data(), n);
        make_twiddles_soa(Wr.data(), Wi.data(), n);

        aos = src;
        split(src.data(), re.data(), im.data(), n);

        const double aos_s = ns_per_sample(
            [&] {
                fft_aos_r2_scalar(aos.data(), n, Dir::Fwd, W.data());
                keep(aos);
            },
            n);
        const double soa_s_native = ns_per_sample(
            [&] {
                fft_soa_r2_scalar(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                keep(re);
            },
            n);

        const double aos_av = ns_per_sample(
            [&] {
                fft_aos_r2_autovec(aos.data(), n, Dir::Fwd, W.data());
                keep(aos);
            },
            n);
        const double soa_av = ns_per_sample(
            [&] {
                fft_soa_r2_autovec(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                keep(re);
            },
            n);
        const double aos_n = ns_per_sample(
            [&] {
                fft_aos_r2_neon(aos.data(), n, Dir::Fwd, W.data());
                keep(aos);
            },
            n);
        const double soa_n = ns_per_sample(
            [&] {
                fft_soa_r2_neon(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                keep(re);
            },
            n);
        const double cvt = ns_per_sample(
            [&] {
                split(src.data(), re.data(), im.data(), n);
                fft_soa_r2_neon(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                join(re.data(), im.data(), aos.data(), n);
                keep(aos);
            },
            n);

        std::printf("%8zu %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %12.3f\n", n, aos_s,
                    soa_s_native, aos_av, soa_av, aos_n, soa_n, cvt);

        double r4_aos_s = 0, r4_soa_s = 0, r4_aos_n = 0, r4_soa_n = 0, r4_cvt = 0;
        if (n % 4 == 0) {
            r4_aos_s = ns_per_sample(
                [&] {
                    fft_aos_r4_scalar(aos.data(), n, Dir::Fwd, W.data());
                    keep(aos);
                },
                n);
            r4_soa_s = ns_per_sample(
                [&] {
                    fft_soa_r4_scalar(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                    keep(re);
                },
                n);
            r4_aos_n = ns_per_sample(
                [&] {
                    fft_aos_r4_neon(aos.data(), n, Dir::Fwd, W.data());
                    keep(aos);
                },
                n);
            r4_soa_n = ns_per_sample(
                [&] {
                    fft_soa_r4_neon(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                    keep(re);
                },
                n);
            r4_cvt = ns_per_sample(
                [&] {
                    split(src.data(), re.data(), im.data(), n);
                    fft_soa_r4_neon(re.data(), im.data(), n, Dir::Fwd, Wr.data(), Wi.data());
                    join(re.data(), im.data(), aos.data(), n);
                    keep(aos);
                },
                n);
        }

        rows.push_back({n, aos_s, soa_s_native, aos_av, soa_av, aos_n, soa_n, cvt, r4_aos_s,
                        r4_soa_s, r4_aos_n, r4_soa_n, r4_cvt});
    }

    std::printf("\nradix-4\n");
    std::printf("%8s %10s %10s %10s %10s %12s\n", "N", "AoS sc", "SoA sc", "AoS NEON", "SoA NEON",
                "cvt+SoA N");
    for (const auto& r : rows) {
        std::printf("%8zu %10.3f %10.3f %10.3f %10.3f %12.3f\n", r.n, r.r4_aos_s, r.r4_soa_s,
                    r.r4_aos_n, r.r4_soa_n, r.r4_cvt);
    }

    std::printf("\n--- csv r2 ---\n");
    std::printf("n,aos_scalar,soa_scalar,aos_autovec,soa_autovec,aos_neon,soa_neon,cvt_soa_neon\n");
    for (const auto& r : rows) {
        std::printf("%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", r.n, r.aos_s, r.soa_s, r.aos_av,
                    r.soa_av, r.aos_n, r.soa_n, r.cvt);
    }
    std::printf("--- csv r4 ---\n");
    std::printf("n,aos_scalar,soa_scalar,aos_neon,soa_neon,cvt_soa_neon\n");
    for (const auto& r : rows) {
        std::printf("%zu,%.4f,%.4f,%.4f,%.4f,%.4f\n", r.n, r.r4_aos_s, r.r4_soa_s, r.r4_aos_n,
                    r.r4_soa_n, r.r4_cvt);
    }
    return 0;
}
