#include "fft/gpu/detail/hierarchical.h"

#include "fft/core/bitops.h"
#include "fft/gpu/detail/cuda_utils.cuh"

#include <cassert>
#include <cmath>

namespace FFTGpu {
namespace detail {
namespace {

constexpr int kMaxSharedBlock = 1024;
constexpr int kMaxWarpFft = 32;

__global__ void shared_block_fft_kernel(double2* __restrict__ data, int n) {
    extern __shared__ double2 s[];
    const int tid = threadIdx.x;
    if (tid >= n) {
        return;
    }

    int bits = 0;
    for (int t = n; t > 1; t >>= 1) {
        ++bits;
    }

    double2* row = data + static_cast<std::size_t>(blockIdx.x) * n;
    s[bit_reverse_device(tid, bits)] = row[tid];
    __syncthreads();

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        if ((tid & (len - 1)) < half) {
            const int i0 = tid;
            const int i1 = tid + half;
            const int j = tid & (half - 1);
            const double angle =
                -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
            const double2 w = make_double2(cos(angle), sin(angle));
            const double2 u = s[i0];
            const double2 t = cmul(w, s[i1]);
            s[i0] = cadd(u, t);
            s[i1] = csub(u, t);
        }
        __syncthreads();
    }

    row[tid] = s[tid];
}

__global__ void warp_fft_kernel(double2* __restrict__ data, int n) {
    const int lane = threadIdx.x;
    const unsigned mask = 0xffffffffu;
    double2* row = data + static_cast<std::size_t>(blockIdx.x) * n;

    double2 v = (lane < n) ? row[lane] : make_double2(0.0, 0.0);

    int bits = 0;
    for (int t = n; t > 1; t >>= 1) {
        ++bits;
    }

    // Bit-reverse: lane L receives row[bitrev(L)]. All lanes must call __shfl_sync.
    {
        const int src = (lane < n) ? bit_reverse_device(lane, bits) : lane;
        v = make_double2(
            __shfl_sync(mask, v.x, src),
            __shfl_sync(mask, v.y, src));
    }

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int j = lane & (half - 1);
        const int in_lower = (lane & (len - 1)) < half;

        const double angle =
            -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
        const double2 w = make_double2(cos(angle), sin(angle));

        const int partner = lane ^ half;
        const double2 other = make_double2(
            __shfl_sync(mask, v.x, partner),
            __shfl_sync(mask, v.y, partner));

        if (lane < n) {
            if (in_lower) {
                const double2 t = cmul(w, other);
                v = cadd(v, t);
            } else {
                const double2 t = cmul(w, v);
                v = csub(other, t);
            }
        }
    }

    if (lane < n) {
        row[lane] = v;
    }
}

__global__ void twiddle_kernel(
    double2* __restrict__ data,
    int count,
    int n1,
    int n2) {
    const int total = count * n1 * n2;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int n = n1 * n2;
    const int local = tid % n;
    const int row = local / n2;
    const int col = local % n2;
    if (row == 0 || col == 0) {
        return;
    }

    const double angle =
        -2.0 * M_PI * static_cast<double>(row * col) / static_cast<double>(n);
    data[tid] = cmul(data[tid], make_double2(cos(angle), sin(angle)));
}

// src: count matrices of n_rows x n_cols → dst: n_cols x n_rows
__global__ void transpose_kernel(
    const double2* __restrict__ src,
    double2* __restrict__ dst,
    int count,
    int n_rows,
    int n_cols) {
    const int total = count * n_rows * n_cols;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int n = n_rows * n_cols;
    const int batch = tid / n;
    const int local = tid % n;
    const int row = local / n_cols;
    const int col = local % n_cols;
    dst[batch * n + col * n_rows + row] = src[batch * n + row * n_cols + col];
}

void launch_leaf(double2* data, int count, int n, FftLeaf leaf) {
    if (n <= 1 || count <= 0) {
        return;
    }

    if (leaf == FftLeaf::SharedBlock) {
        const size_t smem = static_cast<size_t>(n) * sizeof(double2);
        shared_block_fft_kernel<<<count, n, smem>>>(data, n);
    } else {
        warp_fft_kernel<<<count, kMaxWarpFft>>>(data, n);
    }
    FFT_CUDA_CHECK(cudaGetLastError());
}

int leaf_max(FftLeaf leaf) {
    return leaf == FftLeaf::SharedBlock ? kMaxSharedBlock : kMaxWarpFft;
}

// Bailey six-step FFT over `count` contiguous length-`n` vectors in d_a.
// d_b is scratch of the same total length. Result always left in d_a.
void fft_many(double2* d_a, double2* d_b, int count, int n, FftLeaf leaf) {
    if (n <= 1 || count <= 0) {
        return;
    }

    if (n <= leaf_max(leaf)) {
        launch_leaf(d_a, count, n, leaf);
        return;
    }

    const int n1 = choose_four_step_n1(n);
    const int n2 = n / n1;
    constexpr int kThreads = 256;
    const int total = count * n;
    const int blocks = div_ceil(total, kThreads);

    // 1) Transpose each n1 x n2 → n2 x n1
    transpose_kernel<<<blocks, kThreads>>>(d_a, d_b, count, n1, n2);
    FFT_CUDA_CHECK(cudaGetLastError());

    // 2) Column FFTs of the original = row FFTs of length n1
    fft_many(d_b, d_a, count * n2, n1, leaf);

    // 3) Transpose back n2 x n1 → n1 x n2
    transpose_kernel<<<blocks, kThreads>>>(d_b, d_a, count, n2, n1);
    FFT_CUDA_CHECK(cudaGetLastError());

    // 4) Twiddles W^{row*col}
    twiddle_kernel<<<blocks, kThreads>>>(d_a, count, n1, n2);
    FFT_CUDA_CHECK(cudaGetLastError());

    // 5) Row FFTs of length n2
    fft_many(d_a, d_b, count * n1, n2, leaf);

    // 6) Final transpose n1 x n2 → n2 x n1 (natural order)
    transpose_kernel<<<blocks, kThreads>>>(d_a, d_b, count, n1, n2);
    FFT_CUDA_CHECK(cudaGetLastError());

    FFT_CUDA_CHECK(cudaMemcpy(
        d_a,
        d_b,
        static_cast<size_t>(total) * sizeof(double2),
        cudaMemcpyDeviceToDevice));
}

}  // namespace

void fft_hierarchical(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output,
    FftLeaf leaf) {
    const std::size_t n = input.size();
    assert(FFTCore::is_power_of_2(n));
    output.resize(n);

    if (n <= 1) {
        output = input;
        return;
    }

    static_assert(sizeof(FFTCore::Complex) == sizeof(double2),
                  "Complex must match double2 layout");

    DeviceBuffer d_a(n);
    DeviceBuffer d_b(n);
    d_a.upload(input.data());
    fft_many(d_a.get(), d_b.get(), 1, static_cast<int>(n), leaf);
    d_a.download(output.data());
}

}  // namespace detail
}  // namespace FFTGpu
