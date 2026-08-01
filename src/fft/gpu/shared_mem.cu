#include "fft/gpu/shared_mem.h"

#include "fft/core/bitops.h"
#include "fft/gpu/detail/cuda_utils.cuh"
#include "fft/gpu/detail/hierarchical.h"

#include <cassert>
#include <cmath>

namespace {

constexpr int kMaxSharedBlock = 1024;

// Stockham DIF stage — natural order, no separate bit-reversal.
__global__ void stockham_dif_stage_kernel(
    const double2* __restrict__ in,
    double2* __restrict__ out,
    int n,
    int len) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int butterflies = n >> 1;
    if (tid >= butterflies) {
        return;
    }

    const int half = len >> 1;
    const int j = tid % half;
    const int block = tid / half;
    const int i0 = block * len + j;
    const int i1 = i0 + half;

    const double angle =
        -2.0 * M_PI * static_cast<double>(j) / static_cast<double>(len);
    const double2 w = make_double2(cos(angle), sin(angle));

    const double2 u = in[i0];
    const double2 v = in[i1];
    out[block * half + j] = FFTGpu::detail::cadd(u, v);
    out[block * half + j + butterflies] =
        FFTGpu::detail::cmul(FFTGpu::detail::csub(u, v), w);
}

void fft_stockham_device(double2* d_a, double2* d_b, int n) {
    constexpr int kThreads = 256;
    const int stage_blocks = FFTGpu::detail::div_ceil(n / 2, kThreads);

    double2* in = d_a;
    double2* out = d_b;
    for (int len = n; len >= 2; len >>= 1) {
        stockham_dif_stage_kernel<<<stage_blocks, kThreads>>>(in, out, n, len);
        FFT_CUDA_CHECK(cudaGetLastError());
        double2* tmp = in;
        in = out;
        out = tmp;
    }

    if (in != d_a) {
        FFT_CUDA_CHECK(cudaMemcpy(
            d_a,
            in,
            static_cast<size_t>(n) * sizeof(double2),
            cudaMemcpyDeviceToDevice));
    }
}

}  // namespace

void fft_gpu_shared_mem(const std::vector<FFTCore::Complex>& input,
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

    // Block-fitting sizes: entire FFT in __shared__.
    if (n <= static_cast<std::size_t>(kMaxSharedBlock)) {
        FFTGpu::detail::fft_hierarchical(
            input,
            output,
            FFTGpu::detail::FftLeaf::SharedBlock);
        return;
    }

    // Large N: Stockham DIF global-memory stages (distinct from Bailey six-step).
    FFTGpu::detail::DeviceBuffer d_a(n);
    FFTGpu::detail::DeviceBuffer d_b(n);
    d_a.upload(input.data());
    fft_stockham_device(d_a.get(), d_b.get(), static_cast<int>(n));
    d_a.download(output.data());
}
