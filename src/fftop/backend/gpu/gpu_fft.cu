#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"
#include "fftop/backend/gpu/cooley_tukey.h"
#include "fftop/backend/gpu/gpu_utils.h"
#include "fftop/math/fft_math.h"    // shared digit_reverse — __host__ __device__
#include "fftop/math/integer.h"     // is_power_of

#include <cassert>
#include <cuda_runtime.h>

// Complex elements and device pointers are double2 (same layout as std::complex<double>).
// All kernels use double2* internally; the public API uses Complex* and casts at the boundary.

namespace FFTop {
namespace {

// ── Helpers ──────────────────────────────────────────────────────────────────

constexpr unsigned kBlock = 256;

__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ double2 csub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}
__device__ __forceinline__ double2 cmul(double2 w, double2 x) {
    return make_double2(w.x * x.x - w.y * x.y,
                        w.x * x.y + w.y * x.x);
}
// Multiply by  j =  i  →  (-b, a)
__device__ __forceinline__ double2 mul_j(double2 x) {
    return make_double2(-x.y, x.x);
}
// Multiply by -j = -i  →  ( b, -a)
__device__ __forceinline__ double2 mul_minus_j(double2 x) {
    return make_double2(x.y, -x.x);
}

__device__ __forceinline__
double2 twiddle(unsigned k, unsigned order, int inverse) {
    const double sign  = inverse ? 1.0 : -1.0;
    const double twopi = 6.283185307179586476925286766559;
    const double angle = sign * twopi * double(k) / double(order);
    double s, c;
    sincos(angle, &s, &c);
    return make_double2(c, s);
}

// ── Digit-reversal permute ───────────────────────────────────────────────────

// In-place permute: thread i swaps data[i] with data[rev(i)] when i < rev(i).
// The permutation has disjoint pairs, so no two threads touch the same location.
__global__ void k_digit_reverse_permute(double2* data, unsigned n, unsigned radix) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned j = static_cast<unsigned>(
        FFTop::Math::digit_reverse(i, n, radix));
    if (i < j) {
        double2 tmp = data[i];
        data[i]     = data[j];
        data[j]     = tmp;
    }
}

// ── Radix-2 butterfly stage ──────────────────────────────────────────────────

// Each thread handles one (x0, x1) butterfly pair.
// Mirrors the scalar radix2() kernel in kernel.inl exactly.
__global__ void k_radix2_stage(double2* data, unsigned n, unsigned stride, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    const unsigned group_size = stride * 2;
    const unsigned group      = tid / stride;
    const unsigned pos        = tid % stride;

    const unsigned i0 = group * group_size + pos;
    const unsigned i1 = i0 + stride;

    const double2 a  = data[i0];
    const double2 bw = cmul(twiddle(pos, group_size, inverse), data[i1]);

    data[i0] = cadd(a, bw);
    data[i1] = csub(a, bw);
}

// ── Radix-4 butterfly stage ──────────────────────────────────────────────────

// Each thread handles one 4-element butterfly group.
// Mirrors the scalar radix4() kernel in kernel.inl exactly.
__global__ void k_radix4_stage(double2* data, unsigned n, unsigned stride, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 4) return;

    const unsigned group_size = stride * 4;
    const unsigned group      = tid / stride;
    const unsigned pos        = tid % stride;

    const unsigned i0 = group * group_size + pos;
    const unsigned i1 = i0 + stride;
    const unsigned i2 = i1 + stride;
    const unsigned i3 = i2 + stride;

    const double2 a0 = data[i0];
    const double2 a1 = cmul(twiddle(    pos, group_size, inverse), data[i1]);
    const double2 a2 = cmul(twiddle(2 * pos, group_size, inverse), data[i2]);
    const double2 a3 = cmul(twiddle(3 * pos, group_size, inverse), data[i3]);

    const double2 even_sum  = cadd(a0, a2);
    const double2 even_diff = csub(a0, a2);
    const double2 odd_sum   = cadd(a1, a3);
    const double2 odd_diff  = csub(a1, a3);
    const double2 odd_turn  = inverse ? mul_j(odd_diff) : mul_minus_j(odd_diff);

    data[i0] = cadd(even_sum,  odd_sum);
    data[i1] = cadd(even_diff, odd_turn);
    data[i2] = csub(even_sum,  odd_sum);
    data[i3] = csub(even_diff, odd_turn);
}

// ── Helpers for the traversal ────────────────────────────────────────────────

inline unsigned blocks_for(unsigned n) {
    return (n + kBlock - 1) / kBlock;
}

void launch_digit_reverse_permute(double2* d, unsigned n, unsigned radix) {
    k_digit_reverse_permute<<<blocks_for(n), kBlock>>>(d, n, radix);
    GPU::check_cuda(cudaGetLastError(), "digit_reverse_permute");
}

}  // anonymous namespace

// ── IGPURadix implementations ────────────────────────────────────────────────

namespace GPU {

bool GPURadix2::supports(std::size_t n) const { return is_power_of(n, 2); }
bool GPURadix4::supports(std::size_t n) const { return is_power_of(n, 4); }

void GPURadix2::butterfly_stage(Complex* d_data, std::size_t n,
                                 std::size_t stride, Direction dir) const {
    auto* d = reinterpret_cast<double2*>(d_data);
    const int inv = (dir == Direction::Inverse) ? 1 : 0;
    k_radix2_stage<<<blocks_for(static_cast<unsigned>(n / 2)), kBlock>>>(
        d, static_cast<unsigned>(n), static_cast<unsigned>(stride), inv);
    check_cuda(cudaGetLastError(), "radix2_stage");
}

void GPURadix4::butterfly_stage(Complex* d_data, std::size_t n,
                                 std::size_t stride, Direction dir) const {
    auto* d = reinterpret_cast<double2*>(d_data);
    const int inv = (dir == Direction::Inverse) ? 1 : 0;
    k_radix4_stage<<<blocks_for(static_cast<unsigned>(n / 4)), kBlock>>>(
        d, static_cast<unsigned>(n), static_cast<unsigned>(stride), inv);
    check_cuda(cudaGetLastError(), "radix4_stage");
}

// ── IterativeGPUTraversalStrategy ───────────────────────────────────────────
//
// Mirrors IterativeTraversalStrategy::run() from traversal_strategy.cpp:
//   1. digit-reverse permute (one kernel)
//   2. for each stage: one butterfly kernel
//
// Kernels in the same CUDA stream execute in order, so no explicit sync
// is needed between stages.

void IterativeGPUTraversalStrategy::run(Complex* d_data, std::size_t n,
                                        Direction dir,
                                        const IGPURadix& radix) const {
    if (n <= 1) return;
    assert(radix.supports(n));

    auto* d = reinterpret_cast<double2*>(d_data);
    launch_digit_reverse_permute(d, static_cast<unsigned>(n),
                                 static_cast<unsigned>(radix.radix()));

    for (std::size_t stride = 1; stride < n; stride *= radix.radix())
        radix.butterfly_stage(d_data, n, stride, dir);
}

// ── Stockham pass kernel ─────────────────────────────────────────────────────
//
// One pass of the Stockham auto-sort radix-2 FFT.
// Reads from `in` (strided), writes to `out` (consecutive) — no bit-reversal
// ever needed.  h = current sub-FFT size (1, 2, 4, ..., N/2).
//
// Each thread k writes to out[k]:
//   group = k / (2h)     which pair of size-h sub-FFTs to merge
//   pos   = k % h        position within the sub-FFT
//   half  = (k/h) % 2   0 → sum half, 1 → difference half
//
//   in0 = in[group*h + pos]
//   in1 = in[group*h + pos + N/2]   (partner sub-FFT, always N/2 apart)
//   tw  = twiddle(pos, 2h)
//   out[k] = half==0 ? in0 + tw*in1
//                    : in0 - tw*in1

__global__ void k_stockham_pass(const double2* __restrict__ in,
                                 double2* __restrict__ out,
                                 unsigned n, unsigned h, int inverse) {
    const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    const unsigned L     = h * 2;
    const unsigned group = k / L;
    const unsigned pos   = k % h;
    const unsigned half  = (k / h) & 1u;

    const double2 a  = in[group * h + pos];
    const double2 bw = cmul(twiddle(pos, L, inverse), in[group * h + pos + n / 2]);

    out[k] = (half == 0) ? cadd(a, bw) : csub(a, bw);
}

// ── StockhamGPUTraversalStrategy ────────────────────────────────────────────
//
// Ping-pong between d_data and an internal scratch buffer.  After log2(N)
// passes the result is in whichever buffer last received a write; if that is
// the scratch, one device-to-device copy brings it back to d_data.
// No digit-reversal permute is needed — that is the whole point of Stockham.

void StockhamGPUTraversalStrategy::run(Complex* d_data, std::size_t n,
                                       Direction dir,
                                       const IGPURadix& /*radix*/) const {
    if (n <= 1) return;

    GPU::DeviceBuf scratch(n);

    auto* ping = reinterpret_cast<double2*>(d_data);
    double2* pong = scratch.ptr;

    const int  inv   = (dir == Direction::Inverse) ? 1 : 0;
    const auto un    = static_cast<unsigned>(n);
    int        passes = 0;

    for (unsigned h = 1; h < un; h *= 2, ++passes) {
        k_stockham_pass<<<blocks_for(un), kBlock>>>(ping, pong, un, h, inv);
        check_cuda(cudaGetLastError(), "stockham_pass");
        std::swap(ping, pong);
    }

    // After an odd number of passes the result is in ping (scratch).
    // pong is then the original d_data buffer; copy result back into it.
    if (passes & 1) {
        check_cuda(cudaMemcpy(pong, ping, n * sizeof(double2),
                              cudaMemcpyDeviceToDevice),
                   "stockham D2D copy");
    }
}

}  // namespace GPU

// ── CooleyTukeyGPUBackend::execute ──────────────────────────────────────────

void CooleyTukeyGPUBackend::execute(const FFTPlan& plan,
                                    const Buffer&  input,
                                    Buffer&        output) {
    output.resize(plan.size);
    if (plan.size == 0) return;
    assert(input.size() >= plan.size);

    GPU::DeviceBuf d_buf(plan.size);

    GPU::check_cuda(
        cudaMemcpy(d_buf.ptr, input.data(), plan.size * sizeof(double2),
                   cudaMemcpyHostToDevice),
        "CooleyTukey H2D");

    traversal_->run(reinterpret_cast<Complex*>(d_buf.ptr),
                    plan.size, plan.direction, *radix_);

    // Synchronise so a kernel error is attributed here, not to the memcpy.
    GPU::check_cuda(cudaDeviceSynchronize(), "CooleyTukey kernel");

    GPU::check_cuda(
        cudaMemcpy(output.data(), d_buf.ptr, plan.size * sizeof(double2),
                   cudaMemcpyDeviceToHost),
        "CooleyTukey D2H");
}

}  // namespace FFTop
