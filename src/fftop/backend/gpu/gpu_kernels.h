#pragma once

// Device helpers shared by GPU .cu translation units.

#include <cuda_runtime.h>

namespace FFTop::GPU {

constexpr unsigned kBlock = 256;
constexpr unsigned kTile  = 256;  // max inner FFT that fits in shared memory (ping-pong)

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

// W[k] = cis(-2π k / n). Inverse uses the conjugate.
// __ldg hits the read-only cache; the table is never written on device.
__device__ __forceinline__ double2 load_w(const double2* W, unsigned idx, int inverse) {
    double2 w = __ldg(W + idx);
    if (inverse) w.y = -w.y;
    return w;
}

}  // namespace FFTop::GPU
