// Quick layout microbench: interleaved AoS vs split SoA, scalar vs NEON.
// Kernel: N independent radix-2 butterflies  x0' = a+Wb, x1' = a-Wb.
//
//   clang++ -O3 -std=c++17 playground/aos_vs_soa.cpp -o /tmp/aos_vs_soa && /tmp/aos_vs_soa

#include <arm_neon.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__clang__)
#define NOVEC _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define NOVEC
#endif

struct Complex {
    double re, im;
};

struct AoS {
    std::vector<Complex> a, b, W;
};

struct SoA {
    std::vector<double> ar, ai, br, bi, Wr, Wi;
};

AoS make_aos(std::size_t n) {
    AoS x;
    x.a.resize(n);
    x.b.resize(n);
    x.W.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        x.a[i] = {double(i) * 0.001, double(i) * 0.002};
        x.b[i] = {double(i) * 0.003, double(i) * 0.004};
        const double t = 6.283185307179586 * double(i) / double(n);
        x.W[i]         = {std::cos(t), -std::sin(t)};
    }
    return x;
}

SoA to_soa(const AoS& x) {
    const std::size_t n = x.a.size();
    SoA s;
    s.ar.resize(n);
    s.ai.resize(n);
    s.br.resize(n);
    s.bi.resize(n);
    s.Wr.resize(n);
    s.Wi.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        s.ar[i] = x.a[i].re;
        s.ai[i] = x.a[i].im;
        s.br[i] = x.b[i].re;
        s.bi[i] = x.b[i].im;
        s.Wr[i] = x.W[i].re;
        s.Wi[i] = x.W[i].im;
    }
    return s;
}

template <class T>
inline void keep(T& x) {
    asm volatile("" : "+m"(x)::"memory");
}

inline float64x2_t cmul_aos(float64x2_t w, float64x2_t b) {
    const float64x2_t b_flip = vextq_f64(b, b, 1);
    const float64x2_t wr     = vdupq_laneq_f64(w, 0);
    const float64x2_t wi     = vdupq_laneq_f64(w, 1);
    const float64x2_t t0     = vmulq_f64(wr, b);
    const float64x2_t t1     = vmulq_f64(wi, b_flip);
    const float64x2_t nms    = {-1.0, 1.0};
    return vfmaq_f64(t0, t1, nms);
}

void r2_aos_scalar(Complex* a, Complex* b, const Complex* W, std::size_t n) {
    NOVEC
    for (std::size_t i = 0; i < n; ++i) {
        const double tr = W[i].re * b[i].re - W[i].im * b[i].im;
        const double ti = W[i].re * b[i].im + W[i].im * b[i].re;
        const double ar = a[i].re;
        const double ai = a[i].im;
        a[i].re         = ar + tr;
        a[i].im         = ai + ti;
        b[i].re         = ar - tr;
        b[i].im         = ai - ti;
    }
}

void r2_aos_autovec(Complex* a, Complex* b, const Complex* W, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const double tr = W[i].re * b[i].re - W[i].im * b[i].im;
        const double ti = W[i].re * b[i].im + W[i].im * b[i].re;
        const double ar = a[i].re;
        const double ai = a[i].im;
        a[i].re         = ar + tr;
        a[i].im         = ai + ti;
        b[i].re         = ar - tr;
        b[i].im         = ai - ti;
    }
}

void r2_soa_scalar(double* ar, double* ai, double* br, double* bi, const double* Wr,
                   const double* Wi, std::size_t n) {
    NOVEC
    for (std::size_t i = 0; i < n; ++i) {
        const double tr = Wr[i] * br[i] - Wi[i] * bi[i];
        const double ti = Wr[i] * bi[i] + Wi[i] * br[i];
        const double a_r = ar[i];
        const double a_i = ai[i];
        ar[i]            = a_r + tr;
        ai[i]            = a_i + ti;
        br[i]            = a_r - tr;
        bi[i]            = a_i - ti;
    }
}

void r2_soa_autovec(double* ar, double* ai, double* br, double* bi, const double* Wr,
                    const double* Wi, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const double tr = Wr[i] * br[i] - Wi[i] * bi[i];
        const double ti = Wr[i] * bi[i] + Wi[i] * br[i];
        const double a_r = ar[i];
        const double a_i = ai[i];
        ar[i]            = a_r + tr;
        ai[i]            = a_i + ti;
        br[i]            = a_r - tr;
        bi[i]            = a_i - ti;
    }
}

void r2_aos_neon(Complex* a, Complex* b, const Complex* W, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float64x2_t a0 = vld1q_f64(&a[i].re);
        const float64x2_t t  = cmul_aos(vld1q_f64(&W[i].re), vld1q_f64(&b[i].re));
        vst1q_f64(&a[i].re, vaddq_f64(a0, t));
        vst1q_f64(&b[i].re, vsubq_f64(a0, t));
    }
}

// AoS in memory, unzip two complexes into SoA registers, SoA arithmetic, zip back.
void r2_aos_neon_unzip(Complex* a, Complex* b, const Complex* W, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t a0 = vld1q_f64(&a[i].re);
        const float64x2_t a1 = vld1q_f64(&a[i + 1].re);
        const float64x2_t b0 = vld1q_f64(&b[i].re);
        const float64x2_t b1 = vld1q_f64(&b[i + 1].re);
        const float64x2_t w0 = vld1q_f64(&W[i].re);
        const float64x2_t w1 = vld1q_f64(&W[i + 1].re);

        const float64x2_t ar = vtrn1q_f64(a0, a1);
        const float64x2_t ai = vtrn2q_f64(a0, a1);
        const float64x2_t br = vtrn1q_f64(b0, b1);
        const float64x2_t bi = vtrn2q_f64(b0, b1);
        const float64x2_t wr = vtrn1q_f64(w0, w1);
        const float64x2_t wi = vtrn2q_f64(w0, w1);

        const float64x2_t tr = vfmsq_f64(vmulq_f64(wr, br), wi, bi);
        const float64x2_t ti = vfmaq_f64(vmulq_f64(wr, bi), wi, br);

        vst1q_f64(&a[i].re, vtrn1q_f64(vaddq_f64(ar, tr), vaddq_f64(ai, ti)));
        vst1q_f64(&a[i + 1].re, vtrn2q_f64(vaddq_f64(ar, tr), vaddq_f64(ai, ti)));
        vst1q_f64(&b[i].re, vtrn1q_f64(vsubq_f64(ar, tr), vsubq_f64(ai, ti)));
        vst1q_f64(&b[i + 1].re, vtrn2q_f64(vsubq_f64(ar, tr), vsubq_f64(ai, ti)));
    }
    if (i < n) r2_aos_scalar(a + i, b + i, W + i, n - i);
}

void r2_soa_neon(double* ar, double* ai, double* br, double* bi, const double* Wr,
                 const double* Wi, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const float64x2_t a_r = vld1q_f64(ar + i);
        const float64x2_t a_i = vld1q_f64(ai + i);
        const float64x2_t b_r = vld1q_f64(br + i);
        const float64x2_t b_i = vld1q_f64(bi + i);
        const float64x2_t w_r = vld1q_f64(Wr + i);
        const float64x2_t w_i = vld1q_f64(Wi + i);

        const float64x2_t tr = vfmsq_f64(vmulq_f64(w_r, b_r), w_i, b_i);
        const float64x2_t ti = vfmaq_f64(vmulq_f64(w_r, b_i), w_i, b_r);

        vst1q_f64(ar + i, vaddq_f64(a_r, tr));
        vst1q_f64(ai + i, vaddq_f64(a_i, ti));
        vst1q_f64(br + i, vsubq_f64(a_r, tr));
        vst1q_f64(bi + i, vsubq_f64(a_i, ti));
    }
    if (i < n) r2_soa_scalar(ar + i, ai + i, br + i, bi + i, Wr + i, Wi + i, n - i);
}

void split_aos(const AoS& x, SoA& s) {
    const std::size_t n = x.a.size();
    for (std::size_t i = 0; i < n; ++i) {
        s.ar[i] = x.a[i].re;
        s.ai[i] = x.a[i].im;
        s.br[i] = x.b[i].re;
        s.bi[i] = x.b[i].im;
        s.Wr[i] = x.W[i].re;
        s.Wi[i] = x.W[i].im;
    }
}

using Clock = std::chrono::steady_clock;

template <class Fn>
double ns_per_butterfly(Fn&& fn, std::size_t n) {
    int inner = 1;
    fn();
    for (;;) {
        const auto t0 = Clock::now();
        for (int i = 0; i < inner; ++i) fn();
        const auto t1 = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms >= 20.0 || inner >= 1 << 20) {
            double best = ms;
            for (int r = 0; r < 4; ++r) {
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

int main() {
    std::printf("host: Apple Silicon NEON  (best of 7, ns/butterfly)\n");
    std::printf("kernel: x0 += Wb; x1 -= Wb  (in-place radix-2)\n\n");
    std::printf("%10s  %14s  %14s  %14s  %14s  %14s  %14s\n", "N", "AoS scalar", "SoA scalar",
                "AoS autovec", "SoA autovec", "AoS NEON", "SoA NEON");

    const std::size_t sizes[] = {256, 1024, 4096, 16384, 65536, 1u << 20};
    struct Row {
        std::size_t n;
        double aos_s, soa_s, aos_av, soa_av, aos_n, soa_n, aos_unzip, split;
    };
    std::vector<Row> rows;

    for (std::size_t n : sizes) {
        auto aos0 = make_aos(n);
        auto soa0 = to_soa(aos0);

        AoS aos = aos0;
        SoA soa = soa0;
        keep(aos.a);
        keep(soa.ar);

        const double aos_s = ns_per_butterfly(
            [&] {
                r2_aos_scalar(aos.a.data(), aos.b.data(), aos.W.data(), n);
                keep(aos.a);
            },
            n);
        aos = aos0;

        const double soa_s = ns_per_butterfly(
            [&] {
                r2_soa_scalar(soa.ar.data(), soa.ai.data(), soa.br.data(), soa.bi.data(),
                              soa.Wr.data(), soa.Wi.data(), n);
                keep(soa.ar);
            },
            n);
        soa = soa0;

        const double aos_av = ns_per_butterfly(
            [&] {
                r2_aos_autovec(aos.a.data(), aos.b.data(), aos.W.data(), n);
                keep(aos.a);
            },
            n);
        aos = aos0;

        const double soa_av = ns_per_butterfly(
            [&] {
                r2_soa_autovec(soa.ar.data(), soa.ai.data(), soa.br.data(), soa.bi.data(),
                               soa.Wr.data(), soa.Wi.data(), n);
                keep(soa.ar);
            },
            n);
        soa = soa0;

        const double aos_n = ns_per_butterfly(
            [&] {
                r2_aos_neon(aos.a.data(), aos.b.data(), aos.W.data(), n);
                keep(aos.a);
            },
            n);
        aos = aos0;

        const double soa_n = ns_per_butterfly(
            [&] {
                r2_soa_neon(soa.ar.data(), soa.ai.data(), soa.br.data(), soa.bi.data(),
                            soa.Wr.data(), soa.Wi.data(), n);
                keep(soa.ar);
            },
            n);
        soa = soa0;

        const double aos_unzip = ns_per_butterfly(
            [&] {
                r2_aos_neon_unzip(aos.a.data(), aos.b.data(), aos.W.data(), n);
                keep(aos.a);
            },
            n);
        aos = aos0;

        const double split = ns_per_butterfly(
            [&] {
                split_aos(aos, soa);
                r2_soa_neon(soa.ar.data(), soa.ai.data(), soa.br.data(), soa.bi.data(),
                            soa.Wr.data(), soa.Wi.data(), n);
                keep(soa.ar);
            },
            n);

        std::printf("%10zu  %14.3f  %14.3f  %14.3f  %14.3f  %14.3f  %14.3f\n", n, aos_s, soa_s,
                    aos_av, soa_av, aos_n, soa_n);
        rows.push_back({n, aos_s, soa_s, aos_av, soa_av, aos_n, soa_n, aos_unzip, split});
    }

    std::printf("\n%10s  %18s  %22s\n", "N", "AoS NEON unzip", "split+SoA NEON");
    for (const auto& r : rows) {
        std::printf("%10zu  %18.3f  %22.3f\n", r.n, r.aos_unzip, r.split);
    }

    std::printf("\n--- csv ---\n");
    std::printf("n,aos_scalar,soa_scalar,aos_autovec,soa_autovec,aos_neon,soa_neon,"
                "aos_neon_unzip,split_then_soa_neon\n");
    for (const auto& r : rows) {
        std::printf("%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", r.n, r.aos_s, r.soa_s,
                    r.aos_av, r.soa_av, r.aos_n, r.soa_n, r.aos_unzip, r.split);
    }
    return 0;
}
