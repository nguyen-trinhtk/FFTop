// Naive CUDA FFT: bit-reversal + in-place radix-2 stages in global memory.
// One thread per butterfly; host launches one kernel per stage (implicit sync).

#include "fft/gpu/naive.h"

#include "fft/core/bitops.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

#define FFT_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        const cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

__device__ __forceinline__ int bit_reverse_device(int i, int bits) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// out[bitrev(i)] = in[i]
__global__ void bit_reverse_copy_kernel(const double2* __restrict__ in,
                                        double2* __restrict__ out,
                                        int n,
                                        int bits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[bit_reverse_device(i, bits)] = in[i];
}

// In-place radix-2 stage. Butterflies use disjoint index pairs, so this is race-free.
__global__ void radix2_stage_kernel(double2* __restrict__ data, int n, int len) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int half = len >> 1;
    const int num_butterflies = n >> 1;
    if (tid >= num_butterflies) {
        return;
    }

    const int group = tid / half;
    const int j = tid % half;
    const int i0 = group * len + j;
    const int i1 = i0 + half;

    const double angle = -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
    const double2 w = make_double2(cos(angle), sin(angle));

    const double2 u = data[i0];
    const double2 t = cmul(w, data[i1]);
    data[i0] = make_double2(u.x + t.x, u.y + t.y);
    data[i1] = make_double2(u.x - t.x, u.y - t.y);
}

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

}  // namespace

void fft_gpu_naive(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output) {
    const std::size_t n = input.size();
    assert(FFTCore::is_power_of_2(n));
    output.resize(n);

    if (n <= 1) {
        output = input;
        return;
    }

    // std::complex<double> is layout-compatible with double2.
    static_assert(sizeof(FFTCore::Complex) == sizeof(double2),
                  "Complex must match double2 layout");

    const int N = static_cast<int>(n);
    const int bits = static_cast<int>(FFTCore::log2_floor(n));
    const size_t bytes = n * sizeof(double2);
    constexpr int kThreads = 256;

    double2* d_in = nullptr;
    double2* d_data = nullptr;
    FFT_CUDA_CHECK(cudaMalloc(&d_in, bytes));
    FFT_CUDA_CHECK(cudaMalloc(&d_data, bytes));
    FFT_CUDA_CHECK(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice));

    const int copy_blocks = div_ceil(N, kThreads);
    bit_reverse_copy_kernel<<<copy_blocks, kThreads>>>(d_in, d_data, N, bits);
    FFT_CUDA_CHECK(cudaGetLastError());

    const int stage_blocks = div_ceil(N / 2, kThreads);
    for (int len = 2; len <= N; len <<= 1) {
        radix2_stage_kernel<<<stage_blocks, kThreads>>>(d_data, N, len);
        FFT_CUDA_CHECK(cudaGetLastError());
    }

    FFT_CUDA_CHECK(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    FFT_CUDA_CHECK(cudaFree(d_in));
    FFT_CUDA_CHECK(cudaFree(d_data));
}
