#include "fftop/backend/gpu/gpu_traversal.h"
#include "fftop/backend/gpu/gpu_kernels.h"
#include "fftop/backend/gpu/gpu_utils.h"
#include "fftop/math/fft_math.h"

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

namespace FFTop {
namespace {

using GPU::cadd;
using GPU::cmul;
using GPU::csub;
using GPU::kBlock;
using GPU::kTile;
using GPU::load_w;

// Self-sorting DIT Stockham, one radix-2 stage.
// r = n/2, n/4, ..., 1 and l = 1, 2, ..., n/2 with r*l = n/2.
// Natural-order in and out after log2(n) ping-pong stages; no bit-reversal.
__global__ void k_stockham_stage(const double2* __restrict__ in, double2* __restrict__ out,
                                 unsigned n, unsigned r, unsigned log_r, unsigned l,
                                 const double2* __restrict__ W, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    const unsigned k  = tid & (r - 1);
    const unsigned j  = tid >> log_r;
    const unsigned i0 = (j << (log_r + 1)) + k;
    const unsigned o0 = (j << log_r) + k;

    const double2 u = in[i0];
    const double2 t = cmul(load_w(W, j * (n / (l << 1)), inverse), in[i0 + r]);
    out[o0]         = cadd(u, t);
    out[o0 + n / 2] = csub(u, t);
}

// Packed batch of independent n_fft-point Stockham FFTs in shared memory.
// Each FFT is contiguous; ffts_per_block of them share a block.
__global__ void k_stockham_shared_batch(double2* __restrict__ data, unsigned n_fft,
                                        unsigned log_n_fft, unsigned ffts_per_block,
                                        unsigned batch, const double2* __restrict__ W,
                                        unsigned n_full, int inverse) {
    extern __shared__ double2 smem[];

    const unsigned local        = threadIdx.x;
    const unsigned fft_in_block = local >> log_n_fft;
    const unsigned tid          = local & (n_fft - 1);
    const unsigned fft_index    = blockIdx.x * ffts_per_block + fft_in_block;
    const bool     active       = fft_index < batch;

    double2* buf0 = smem + (size_t)fft_in_block * (n_fft << 1);
    double2* buf1 = buf0 + n_fft;

    if (active) buf0[tid] = data[(size_t)fft_index * n_fft + tid];
    __syncthreads();

    double2* src = buf0;
    double2* dst = buf1;
    unsigned r   = n_fft;
    unsigned l   = 1;
    while (r > 1) {
        r >>= 1;
        if (active && tid < (n_fft >> 1)) {
            const unsigned lr   = __ffs(r) - 1;
            const unsigned k    = tid & (r - 1);
            const unsigned j    = tid >> lr;
            const unsigned i0   = (j << (lr + 1)) + k;
            const unsigned o0   = (j << lr) + k;
            const unsigned widx = j * (n_full / (l << 1));
            const double2  u    = src[i0];
            const double2  t    = cmul(load_w(W, widx, inverse), src[i0 + r]);
            dst[o0]                = cadd(u, t);
            dst[o0 + (n_fft >> 1)] = csub(u, t);
        }
        __syncthreads();
        l <<= 1;
        double2* tmp = src;
        src          = dst;
        dst          = tmp;
    }

    if (active) data[(size_t)fft_index * n_fft + tid] = src[tid];
}

constexpr int kTX = 16;

// Column-major transpose of `batch` matrices, each rows x cols -> cols x rows.
__global__ void k_transpose(const double2* __restrict__ in, double2* __restrict__ out,
                            unsigned rows, unsigned cols) {
    __shared__ double2 tile[kTX][kTX + 1];

    const unsigned col = blockIdx.x * kTX + threadIdx.x;
    const unsigned row = blockIdx.y * kTX + threadIdx.y;
    const unsigned b   = blockIdx.z;
    const size_t   mat = (size_t)rows * cols;
    const double2* src = in + (size_t)b * mat;
    double2*       dst = out + (size_t)b * mat;

    if (col < cols && row < rows)
        tile[threadIdx.y][threadIdx.x] = src[(size_t)row + (size_t)rows * col];
    __syncthreads();

    const unsigned d_row = blockIdx.x * kTX + threadIdx.y;
    const unsigned d_col = blockIdx.y * kTX + threadIdx.x;
    if (d_row < cols && d_col < rows)
        dst[(size_t)d_row + (size_t)cols * d_col] = tile[threadIdx.x][threadIdx.y];
}

__global__ void k_twiddle_2d(double2* __restrict__ data, unsigned N1, unsigned N2,
                             unsigned fft_size, const double2* __restrict__ W, unsigned n_full,
                             int inverse) {
    const unsigned k2 = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n1 = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned b  = blockIdx.z;
    if (k2 >= N2 || n1 >= N1) return;

    // After the N2-FFT + layout t[k2 + N2*n1]: W_M^{n1 k2} with M = fft_size.
    const unsigned widx = (unsigned)((unsigned long long)n1 * k2 * (n_full / fft_size));
    double2        z    = data[(size_t)b * fft_size + k2 + (size_t)N2 * n1];
    data[(size_t)b * fft_size + k2 + (size_t)N2 * n1] = cmul(load_w(W, widx, inverse), z);
}

void launch_shared_batch(double2* data, unsigned n_fft, unsigned batch, const double2* W,
                         unsigned n_full, int inv) {
    if (n_fft <= 1 || batch == 0) return;
    const unsigned log_n = GPU::log2_floor(n_fft);
    unsigned ffts_per_block = kBlock / n_fft;
    if (ffts_per_block == 0) ffts_per_block = 1;
    const unsigned threads = ffts_per_block * n_fft;
    const unsigned blocks  = GPU::blocks_for(batch, ffts_per_block);
    const size_t   smem    = (size_t)ffts_per_block * (n_fft << 1) * sizeof(double2);
    k_stockham_shared_batch<<<blocks, threads, smem>>>(data, n_fft, log_n, ffts_per_block, batch,
                                                       W, n_full, inv);
    GPU::check_cuda(cudaGetLastError(), "stockham_shared_batch");
}

constexpr unsigned kMaxGridZ = 65535;

void launch_transpose(const double2* in, double2* out, unsigned rows, unsigned cols,
                      unsigned batch) {
    if (rows == 0 || cols == 0 || batch == 0) return;
    dim3 block(kTX, kTX);
    const size_t mat = (size_t)rows * cols;
    for (unsigned off = 0; off < batch; off += kMaxGridZ) {
        const unsigned chunk = batch - off < kMaxGridZ ? batch - off : kMaxGridZ;
        dim3 grid(GPU::blocks_for(cols, kTX), GPU::blocks_for(rows, kTX), chunk);
        k_transpose<<<grid, block>>>(in + (size_t)off * mat, out + (size_t)off * mat, rows, cols);
        GPU::check_cuda(cudaGetLastError(), "transpose");
    }
}

void launch_twiddle_2d(double2* data, unsigned N1, unsigned N2, unsigned fft_size, unsigned batch,
                       const double2* W, unsigned n_full, int inv) {
    dim3 block(16, 16);
    for (unsigned off = 0; off < batch; off += kMaxGridZ) {
        const unsigned chunk = batch - off < kMaxGridZ ? batch - off : kMaxGridZ;
        dim3 grid(GPU::blocks_for(N2, 16), GPU::blocks_for(N1, 16), chunk);
        k_twiddle_2d<<<grid, block>>>(data + (size_t)off * fft_size, N1, N2, fft_size, W, n_full,
                                      inv);
        GPU::check_cuda(cudaGetLastError(), "twiddle_2d");
    }
}

// Hierarchical four-step. Input in `a`, scratch `b`. Returns the buffer that
// holds `batch` packed FFTs of length `fft_size`.
double2* fft_tiled(double2* a, double2* b, unsigned fft_size, unsigned batch, const double2* W,
                   unsigned n_full, int inv) {
    if (fft_size <= 1 || batch == 0) return a;
    if (fft_size <= kTile) {
        launch_shared_batch(a, fft_size, batch, W, n_full, inv);
        return a;
    }

    const unsigned N2 = kTile;
    const unsigned N1 = fft_size / N2;
    assert(N1 * N2 == fft_size);

    launch_transpose(a, b, N1, N2, batch);
    fft_tiled(b, a, N2, batch * N1, W, n_full, inv);
    launch_twiddle_2d(b, N1, N2, fft_size, batch, W, n_full, inv);
    launch_transpose(b, a, N2, N1, batch);

    double2* inner = fft_tiled(a, b, N1, batch * N2, W, n_full, inv);
    if (inner == a) {
        launch_transpose(a, b, N1, N2, batch);
        return b;
    }
    launch_transpose(b, a, N1, N2, batch);
    return a;
}

std::uint64_t tiled_bytes(unsigned fft_size, unsigned batch) {
    if (fft_size <= 1 || batch == 0) return 0;
    const std::uint64_t rw = 2ull * fft_size * batch * sizeof(double2);
    if (fft_size <= kTile) return rw;
    const unsigned N2 = kTile;
    const unsigned N1 = fft_size / N2;
    return rw + tiled_bytes(N2, batch * N1) + rw + rw + tiled_bytes(N1, batch * N2) + rw;
}

}  // namespace

namespace GPU {

Complex* StockhamGlobalTraversalStrategy::run(Complex* a, Complex* b, const Complex* W,
                                              std::size_t n, Direction dir,
                                              const IGPURadix&) const {
    if (n <= 1) return a;
    assert(Math::is_power_of(n, 2));

    auto* src = reinterpret_cast<double2*>(a);
    auto* dst = reinterpret_cast<double2*>(b);
    auto* Wt  = reinterpret_cast<const double2*>(W);
    const unsigned nu  = static_cast<unsigned>(n);
    const int      inv = (dir == Direction::Inverse) ? 1 : 0;

    unsigned r = nu;
    unsigned l = 1;
    while (r > 1) {
        r >>= 1;
        k_stockham_stage<<<blocks_for(nu / 2), kBlock>>>(src, dst, nu, r, log2_floor(r), l, Wt,
                                                         inv);
        check_cuda(cudaGetLastError(), "stockham_global_stage");
        l <<= 1;
        double2* tmp = src;
        src          = dst;
        dst          = tmp;
    }
    return reinterpret_cast<Complex*>(src);
}

std::uint64_t StockhamGlobalTraversalStrategy::estimated_global_bytes(std::size_t n,
                                                                      std::size_t) const {
    if (n <= 1) return 0;
    const std::uint64_t stages = log2_floor(static_cast<unsigned>(n));
    return stages * 2ull * n * sizeof(double2);
}

Complex* StockhamSharedTraversalStrategy::run(Complex* a, Complex* b, const Complex* W,
                                              std::size_t n, Direction dir,
                                              const IGPURadix&) const {
    if (n <= 1) return a;
    assert(Math::is_power_of(n, 2));
    auto* da = reinterpret_cast<double2*>(a);
    auto* db = reinterpret_cast<double2*>(b);
    auto* Wt = reinterpret_cast<const double2*>(W);
    const int inv = (dir == Direction::Inverse) ? 1 : 0;
    double2* out =
        fft_tiled(da, db, static_cast<unsigned>(n), 1, Wt, static_cast<unsigned>(n), inv);
    return reinterpret_cast<Complex*>(out);
}

std::uint64_t StockhamSharedTraversalStrategy::estimated_global_bytes(std::size_t n,
                                                                      std::size_t) const {
    return tiled_bytes(static_cast<unsigned>(n), 1);
}

}  // namespace GPU
}  // namespace FFTop
