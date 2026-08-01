#include "fft/gpu/detail/iterative.h"

#include "fft/core/bitops.h"
#include "fft/gpu/detail/cuda_utils.cuh"

#include <cassert>
#include <cmath>

namespace FFTGpu {
namespace detail {
namespace {

constexpr int kTile = 1024;
constexpr int kWarp = 32;
constexpr int kThreads = 256;

__global__ void bit_reverse_copy_kernel(
    const double2* __restrict__ in,
    double2* __restrict__ out,
    int n,
    int bits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[bit_reverse_device(i, bits)] = in[i];
}

__global__ void radix2_global_stage_kernel(
    double2* __restrict__ data,
    int n,
    int len) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = len >> 1;
    if (tid >= (n >> 1)) {
        return;
    }

    const int group = tid / half;
    const int j = tid % half;
    const int i0 = group * len + j;
    const int i1 = i0 + half;

    const double angle =
        -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
    const double2 w = make_double2(cos(angle), sin(angle));

    const double2 u = data[i0];
    const double2 t = cmul(w, data[i1]);
    data[i0] = cadd(u, t);
    data[i1] = csub(u, t);
}

// CTA owns `tile` contiguous bit-reversed points; DIT stages [start_len, tile]
// stay in __shared__ (one global load + store per tile).
__global__ void shared_fused_stages_kernel(
    double2* __restrict__ data,
    int tile,
    int start_len) {
    extern __shared__ double2 s[];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * tile;

    s[tid] = data[base + tid];
    __syncthreads();

    for (int len = start_len; len <= tile; len <<= 1) {
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

    data[base + tid] = s[tid];
}

// One warp (32 threads) owns up to 32 contiguous points; stages via __shfl_*.
__global__ void warp_fused_stages_kernel(double2* __restrict__ data, int n) {
    const int lane = threadIdx.x;
    const unsigned mask = 0xffffffffu;
    const int base = blockIdx.x * kWarp;
    const int max_len = ((n - base) >= kWarp) ? kWarp : (n - base);

    double2 v =
        (lane < max_len) ? data[base + lane] : make_double2(0.0, 0.0);

    for (int len = 2; len <= max_len; len <<= 1) {
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

        if (lane < max_len) {
            if (in_lower) {
                v = cadd(v, cmul(w, other));
            } else {
                v = csub(other, cmul(w, v));
            }
        }
    }

    if (lane < max_len) {
        data[base + lane] = v;
    }
}

void run_global_stages(double2* data, int n, int first_len) {
    const int stage_blocks = div_ceil(n / 2, kThreads);
    for (int len = first_len; len <= n; len <<= 1) {
        radix2_global_stage_kernel<<<stage_blocks, kThreads>>>(data, n, len);
        FFT_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace

void fft_iterative_locality(
    const std::vector<FFTCore::Complex>& input,
    std::vector<FFTCore::Complex>& output,
    LocalityMode mode) {
    const std::size_t n = input.size();
    assert(FFTCore::is_power_of_2(n));
    output.resize(n);

    if (n <= 1) {
        output = input;
        return;
    }

    static_assert(sizeof(FFTCore::Complex) == sizeof(double2),
                  "Complex must match double2 layout");

    const int N = static_cast<int>(n);
    const int bits = static_cast<int>(FFTCore::log2_floor(n));

    DeviceBuffer d_in(n);
    DeviceBuffer d_data(n);
    d_in.upload(input.data());

    bit_reverse_copy_kernel<<<div_ceil(N, kThreads), kThreads>>>(
        d_in.get(), d_data.get(), N, bits);
    FFT_CUDA_CHECK(cudaGetLastError());

    if (mode == LocalityMode::GlobalOnly) {
        run_global_stages(d_data.get(), N, 2);
        d_data.download(output.data());
        return;
    }

    int stages_done_through = 1;

    // WarpThenShared (optimized)
    {
        const int warp_blocks = (N >= kWarp) ? (N / kWarp) : 1;
        warp_fused_stages_kernel<<<warp_blocks, kWarp>>>(d_data.get(), N);
        FFT_CUDA_CHECK(cudaGetLastError());
        stages_done_through = (N >= kWarp) ? kWarp : N;

        const int tile = (N < kTile) ? N : kTile;
        if (tile > stages_done_through) {
            const size_t smem = static_cast<size_t>(tile) * sizeof(double2);
            shared_fused_stages_kernel<<<N / tile, tile, smem>>>(
                d_data.get(), tile, stages_done_through << 1);
            FFT_CUDA_CHECK(cudaGetLastError());
            stages_done_through = tile;
        }
    }

    const int first_global = stages_done_through << 1;
    if (first_global <= N) {
        run_global_stages(d_data.get(), N, first_global);
    }

    d_data.download(output.data());
}

}  // namespace detail
}  // namespace FFTGpu
