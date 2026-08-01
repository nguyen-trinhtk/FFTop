#include "fft/gpu/naive.h"

#include "fft/core/bitops.h"
#include "fft/gpu/detail/cuda_utils.cuh"

#include <cassert>
#include <cmath>

namespace {

__global__ void bit_reverse_copy_kernel(const double2* __restrict__ in,
                                        double2* __restrict__ out,
                                        int n,
                                        int bits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[FFTGpu::detail::bit_reverse_device(i, bits)] = in[i];
}

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

    const double angle =
        -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
    const double2 w = make_double2(cos(angle), sin(angle));

    const double2 u = data[i0];
    const double2 t = FFTGpu::detail::cmul(w, data[i1]);
    data[i0] = FFTGpu::detail::cadd(u, t);
    data[i1] = FFTGpu::detail::csub(u, t);
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

    static_assert(sizeof(FFTCore::Complex) == sizeof(double2),
                  "Complex must match double2 layout");

    const int N = static_cast<int>(n);
    const int bits = static_cast<int>(FFTCore::log2_floor(n));
    constexpr int kThreads = 256;

    FFTGpu::detail::DeviceBuffer d_in(n);
    FFTGpu::detail::DeviceBuffer d_data(n);
    d_in.upload(input.data());

    const int copy_blocks = FFTGpu::detail::div_ceil(N, kThreads);
    bit_reverse_copy_kernel<<<copy_blocks, kThreads>>>(
        d_in.get(), d_data.get(), N, bits);
    FFT_CUDA_CHECK(cudaGetLastError());

    const int stage_blocks = FFTGpu::detail::div_ceil(N / 2, kThreads);
    for (int len = 2; len <= N; len <<= 1) {
        radix2_stage_kernel<<<stage_blocks, kThreads>>>(d_data.get(), N, len);
        FFT_CUDA_CHECK(cudaGetLastError());
    }

    d_data.download(output.data());
}
