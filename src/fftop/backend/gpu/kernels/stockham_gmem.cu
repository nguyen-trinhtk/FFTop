// Stockham gmem: self-sorting, one radix-2 global-memory kernel per stage, ping-pong a/b.

#include "fftop/backend/gpu/kernels/common.cuh"

#include <utility>

namespace FFTop::GPU {
namespace {

/// @brief Run one stage of DIT Stockham
/// @param out Ptr to output buffer (non-overlapping)
/// @param in Ptr to input buffer (non-overlapping)
/// @param n FFT size
/// @param i Stage index (from 0 to log2(n)-1)
/// @param Ws Ptr to twiddles of stage
/// @param inverse Inverse or not?
__global__ void k_stockham_stage(double2* __restrict__ out, const double2* __restrict__ in,
    unsigned n, unsigned i, const double2* __restrict__ Ws, int inverse) {

    const unsigned half = n / 2;
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= half)
        return;

    // Split tid's bits
    const unsigned s = (1u << i); // stride
    const unsigned lo = tid & (s - 1); // tid % stride
    const unsigned hi = tid >> i; // i0/stride

    // Input/output indices
    const unsigned i0 = tid;
    const unsigned i1 = tid + half;
    const unsigned o0 = (hi << (i + 1)) + lo;
    const unsigned o1 = o0 + s;

    // Load twiddle
    const double2 w = load_w(Ws, lo, inverse);
    // Butterfly
    const double2 a = in[i0];
    const double2 wb = cmul(w, in[i1]);
    out[o0] = cadd(a, wb);
    out[o1] = csub(a, wb);
}
}  // namespace

Complex* stockham_gmem(Complex* a, Complex* b, const Complex* W, unsigned n, Direction dir,
                       unsigned block) {
    const int inv = is_inverse(dir);
    unsigned log_n = Math::log2_floor(n);
    for (unsigned i = 0; i < log_n; i++) {
        const unsigned s = 1u << i;
        k_stockham_stage<<<fill_grid(n / 2, block), block>>>(as_d2(b), as_d2(a), n, i, as_d2(W) + (s - 1), inv);
        check_cuda(cudaGetLastError(), "stockham_gmem_stage");
        std::swap(a, b);
    }
    return a;
}
}  // namespace FFTop::GPU
