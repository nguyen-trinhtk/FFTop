#pragma once

#include "fftop/backend/gpu/gpu.h"
#include "fftop/math/fft_math.h"
#include <cuda_runtime.h>

namespace FFTop::GPU {
// double2 <-> complex
inline double2* as_d2(Complex* p) { return reinterpret_cast<double2*>(p); }
inline const double2* as_d2(const Complex* p) { return reinterpret_cast<const double2*>(p); }
inline Complex* as_complex(double2* p) { return reinterpret_cast<Complex*>(p); }
inline int is_inverse(Direction dir) { return dir == Direction::Inverse ? 1 : 0; }

inline unsigned fill_grid(unsigned n, unsigned block) {
    return static_cast<unsigned>(Math::ceil_div(n, block));
}

__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ double2 csub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ double2 cmul(double2 w, double2 x) {
    return make_double2(w.x * x.x - w.y * x.y, w.x * x.y + w.y * x.x);
}
__device__ __forceinline__ double2 mul_j(double2 x) { return make_double2(-x.y, x.x); }
__device__ __forceinline__ double2 mul_minus_j(double2 x) { return make_double2(x.y, -x.x); }


/// @brief Load twiddle from twiddle table
/// @param W pointer to twiddle table
/// @param idx index of twiddle in twiddle table
/// @param inverse inverse FFT or not
/// @return twiddle for idx (conjugated if inverse)
__device__ __forceinline__ double2 load_w(const double2* W, unsigned idx, int inverse) {
    double2 w = __ldg(W + idx); // load from read-only cache
    if (inverse)
        w.y = -w.y;
    return w;
}
}  // namespace FFTop::GPU
