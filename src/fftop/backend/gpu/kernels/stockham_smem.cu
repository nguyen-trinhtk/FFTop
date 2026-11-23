// Stockham smem: split N into a few power-of-2 factors and do one pass per factor.
// Each pass runs small column FFTs in shared memory; the last one leaves the output in order.

#include "fftop/backend/gpu/kernels/common.cuh"

#include <algorithm>
#include <utility>

#define GPU_UNROLL // Put a placeholder for now, TODO: unroll

namespace FFTop::GPU {
namespace {

/// @brief One FFT pass: a block takes 8 columns, FFTs them in smem, and writes them where the
///        next pass wants them
/// @param out Ptr to output buffer (non-overlapping)
/// @param in Ptr to input buffer (non-overlapping)
/// @param nrows FFT size of this pass (column length)
/// @param ncols Number of columns (N / nrows)
/// @param log2_s log2 of all previous pass sizes multiplied (0 on the first pass)
/// @param W Ptr to per-stage twiddle table
/// @param inverse Inverse or not?
__global__ void k_stockham_pass(
    double2* __restrict__ out, const double2* __restrict__ in, unsigned nrows, unsigned ncols,
    unsigned log2_s, const double2* __restrict__ W, int inverse) {

    const unsigned col = threadIdx.x % GPU_TILE_WIDTH; // column in block
    const unsigned gcol = blockIdx.x * GPU_TILE_WIDTH + col; // column in data
    const unsigned first_row = threadIdx.x / GPU_TILE_WIDTH;
    const unsigned row_step = blockDim.x / GPU_TILE_WIDTH; // num threads per column
    const bool active = gcol < ncols; // ncols < GPU_TILE_WIDTH leaves some columns empty

    // Ping-pong buffers, size set at launch
    extern __shared__ double2 buf[];
    double2* src = buf;
    double2* dst = buf + nrows * GPU_TILE_WIDTH;

    const unsigned stride = 1u << log2_s;
    const unsigned col_lo = gcol & (stride - 1); // gcol % stride
    const unsigned col_hi = gcol >> log2_s; // gcol / stride

    // Load into smem and apply twiddles (all 1 on the first pass)
    if (active) {
        // Table only has the first half of the twiddles, the second half is just the negative
        const unsigned half_sr = stride * nrows / 2;
        GPU_UNROLL
        for (unsigned r = first_row; r < nrows; r += row_step) {
            const unsigned e = col_lo * r;
            double2 w = load_w(W + (half_sr - 1), e & (half_sr - 1), inverse);
            if (e >= half_sr) w = make_double2(-w.x, -w.y);
            src[r * GPU_TILE_WIDTH + col] = cmul(w, in[r * ncols + gcol]);
        }
    }
    __syncthreads();

    // Stockham FFT on columns
    for (unsigned i = 0, s = 1; s < nrows; i++, s *= 2) {
        const unsigned half = nrows / 2;
        GPU_UNROLL
        for (unsigned tid = first_row; tid < half; tid += row_step) {
            // Split tid's bits
            const unsigned lo = tid & (s - 1); // tid % stride
            const unsigned hi = tid >> i; // i0/stride
            // Input/output indices
            const unsigned i0 = tid;
            const unsigned i1 = tid + half;
            const unsigned o0 = (hi << (i + 1)) + lo;
            const unsigned o1 = o0 + s;
            // Load twiddle
            const double2 w = load_w(W + (s - 1), lo, inverse);
            // Butterfly
            const double2 a = src[i0 * GPU_TILE_WIDTH + col];
            const double2 wb = cmul(w, src[i1 * GPU_TILE_WIDTH + col]);
            dst[o0 * GPU_TILE_WIDTH + col] = cadd(a, wb);
            dst[o1 * GPU_TILE_WIDTH + col] = csub(a, wb);
        }
        __syncthreads();
        // Swap src and dst
        double2* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Store src back
    if (!active) 
        return;

    // Write back
    for (unsigned r = first_row; r < nrows; r += row_step) {
        // TODO: first pass writes are strided, transpose via smem
        out[col_lo + stride * (r + nrows * col_hi)] = src[r * GPU_TILE_WIDTH + col];
    }
}
}  // namespace

Complex* stockham_smem(Complex* a, Complex* b, const Complex* W, unsigned n, Direction dir,
                       unsigned passes, unsigned block) {
    const int inv = is_inverse(dir);
    const unsigned log2_n = Math::log2_floor(n);

    for (unsigned p = 0, log2_s = 0; p < passes; p++) {
        // Balanced
        const unsigned log2_rows = log2_n / passes + (p < log2_n % passes ? 1 : 0);
        const unsigned nrows = 1u << log2_rows;
        const unsigned ncols = n >> log2_rows;
        
        unsigned grid = fill_grid(ncols, GPU_TILE_WIDTH);
        block = std::min(block, GPU_TILE_WIDTH * nrows / 2); // one butterfly each
        unsigned smem = GPU_SMEM_ROW_BYTES * nrows;

        // Launch kernel for pass
        k_stockham_pass<<<grid, block, smem>>>(as_d2(b), as_d2(a), nrows, ncols, log2_s, as_d2(W), inv);
        check_cuda(cudaGetLastError(), "stockham_smem pass");
        std::swap(a, b);

        log2_s += log2_rows;
    }
    return a;
}

// Check smem opt in
void stockham_smem_opt_in(std::size_t bytes) {
    check_cuda(cudaFuncSetAttribute(k_stockham_pass, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)), "stockham_smem smem opt-in");
}
}  // namespace FFTop::GPU
