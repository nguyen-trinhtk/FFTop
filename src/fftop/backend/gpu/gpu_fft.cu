#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"
#include "fftop/backend/gpu/gpu_kernels.h"
#include "fftop/backend/gpu/gpu_utils.h"
#include "fftop/math/fft_math.h"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

// Complex elements and device pointers are double2 (same layout as std::complex<double>).
// All kernels use double2* internally; the public API uses Complex* and casts at the boundary.

namespace FFTop {
namespace {

using GPU::cadd;
using GPU::csub;
using GPU::cmul;
using GPU::kBlock;
using GPU::load_w;
using GPU::mul_j;
using GPU::mul_minus_j;

__global__ void k_inplace_dit_reorder(double2* data, unsigned n, unsigned radix) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned j = static_cast<unsigned>(FFTop::Math::digit_reverse(i, n, radix));
    if (i < j) {
        double2 tmp = data[i];
        data[i]     = data[j];
        data[j]     = tmp;
    }
}

// Each thread handles one (x0, x1) butterfly pair.
__global__ void k_radix2_stage(double2* data, unsigned n, unsigned stride, unsigned log_stride,
                               const double2* W, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    const unsigned pos        = tid & (stride - 1);
    const unsigned group      = tid >> log_stride;
    const unsigned group_size = stride << 1;
    const unsigned i0         = (group * group_size) + pos;
    const unsigned i1         = i0 + stride;

    const double2 a  = data[i0];
    const double2 bw = cmul(load_w(W, pos * (n / group_size), inverse), data[i1]);
    data[i0]         = cadd(a, bw);
    data[i1]         = csub(a, bw);
}

// Each thread handles one 4-element butterfly group.
__global__ void k_radix4_stage(double2* data, unsigned n, unsigned stride, unsigned log_stride,
                               const double2* W, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 4) return;

    const unsigned pos        = tid & (stride - 1);
    const unsigned group      = tid >> log_stride;
    const unsigned group_size = stride << 2;
    const unsigned i0         = (group * group_size) + pos;
    const unsigned i1         = i0 + stride;
    const unsigned i2         = i1 + stride;
    const unsigned i3         = i2 + stride;
    const unsigned step       = n / group_size;

    const double2 a0 = data[i0];
    const double2 a1 = cmul(load_w(W, pos * step, inverse), data[i1]);
    const double2 a2 = cmul(load_w(W, (pos << 1) * step, inverse), data[i2]);
    const double2 a3 = cmul(load_w(W, (pos * 3) * step, inverse), data[i3]);

    const double2 even_sum  = cadd(a0, a2);
    const double2 even_diff = csub(a0, a2);
    const double2 odd_sum   = cadd(a1, a3);
    const double2 odd_diff  = csub(a1, a3);
    const double2 odd_turn  = inverse ? mul_j(odd_diff) : mul_minus_j(odd_diff);

    data[i0] = cadd(even_sum, odd_sum);
    data[i1] = cadd(even_diff, odd_turn);
    data[i2] = csub(even_sum, odd_sum);
    data[i3] = csub(even_diff, odd_turn);
}

}  // anonymous namespace

namespace GPU {

bool GPURadix2::supports(std::size_t n) const { return Math::is_power_of(n, 2); }
bool GPURadix4::supports(std::size_t n) const { return Math::is_power_of(n, 4); }

void GPURadix2::butterfly_stage(Complex* d_data, std::size_t n, std::size_t stride, Direction dir,
                                const Complex* d_W) const {
    auto* d  = reinterpret_cast<double2*>(d_data);
    auto* W  = reinterpret_cast<const double2*>(d_W);
    const int inv = (dir == Direction::Inverse) ? 1 : 0;
    const unsigned nu = static_cast<unsigned>(n);
    const unsigned st = static_cast<unsigned>(stride);
    k_radix2_stage<<<blocks_for(nu / 2), kBlock>>>(d, nu, st, log2_floor(st), W, inv);
    check_cuda(cudaGetLastError(), "radix2_stage");
}

void GPURadix4::butterfly_stage(Complex* d_data, std::size_t n, std::size_t stride, Direction dir,
                                const Complex* d_W) const {
    auto* d  = reinterpret_cast<double2*>(d_data);
    auto* W  = reinterpret_cast<const double2*>(d_W);
    const int inv = (dir == Direction::Inverse) ? 1 : 0;
    const unsigned nu = static_cast<unsigned>(n);
    const unsigned st = static_cast<unsigned>(stride);
    k_radix4_stage<<<blocks_for(nu / 4), kBlock>>>(d, nu, st, log2_floor(st), W, inv);
    check_cuda(cudaGetLastError(), "radix4_stage");
}

Complex* IterativeGPUTraversalStrategy::run(Complex* a, Complex* /*b*/, const Complex* W,
                                            std::size_t n, Direction dir,
                                            const IGPURadix& radix) const {
    if (n <= 1) return a;
    assert(radix.supports(n));

    auto* d = reinterpret_cast<double2*>(a);
    k_inplace_dit_reorder<<<blocks_for(static_cast<unsigned>(n)), kBlock>>>(
        d, static_cast<unsigned>(n), static_cast<unsigned>(radix.radix()));
    check_cuda(cudaGetLastError(), "inplace_dit_reorder");

    for (std::size_t stride = 1; stride < n; stride *= radix.radix())
        radix.butterfly_stage(a, n, stride, dir, W);
    return a;
}

std::uint64_t IterativeGPUTraversalStrategy::estimated_global_bytes(std::size_t n,
                                                                    std::size_t radix) const {
    if (n <= 1 || radix < 2) return 0;
    const std::uint64_t stage = 2ull * n * sizeof(double2);
    std::uint64_t stages = 0;
    for (std::size_t stride = 1; stride < n; stride *= radix) ++stages;
    return (1 + stages) * stage;  // bit-reversal + each butterfly stage
}

}  // namespace GPU
}  // namespace FFTop
