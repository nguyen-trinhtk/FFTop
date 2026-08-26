#include "fftop/types.h"
#include "fftop/backend/gpu/gpu_utils.h"

#include <cuda_runtime.h>

namespace FFTop {
namespace {

using GPU::check_cuda;
using GPU::DeviceBuf;

// One output bin per thread: X[k] = sum_t x[t] * exp(sign * 2π i k t / N).
// Inverse is left unnormalised, matching the CPU backends.
__global__ void dft_kernel(const double2* in, double2* out, unsigned n, int inverse) {
    const unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    const double sign  = inverse ? 1.0 : -1.0;
    const double twopi = 6.283185307179586476925286766559;
    double       sum_re = 0.0;
    double       sum_im = 0.0;

    for (unsigned t = 0; t < n; ++t) {
        const double angle = sign * twopi * double(k) * double(t) / double(n);
        double       c, s;
        sincos(angle, &s, &c);
        const double2 x = in[t];
        sum_re += x.x * c - x.y * s;
        sum_im += x.x * s + x.y * c;
    }
    out[k] = make_double2(sum_re, sum_im);
}

}  // namespace

void naive_dft_cuda(const Complex* input, Complex* output, std::size_t n, Direction dir) {
    DeviceBuf d_in(n);
    DeviceBuf d_out(n);

    check_cuda(cudaMemcpy(d_in.ptr, input, n * sizeof(double2), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    constexpr unsigned kThreads = 256;
    const unsigned     blocks   = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    dft_kernel<<<blocks, kThreads>>>(d_in.ptr, d_out.ptr, static_cast<unsigned>(n),
                                     dir == Direction::Inverse ? 1 : 0);
    check_cuda(cudaGetLastError(), "dft_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "dft_kernel");

    check_cuda(cudaMemcpy(output, d_out.ptr, n * sizeof(double2), cudaMemcpyDeviceToHost),"cudaMemcpy D2H");
}

}  // namespace FFTop
