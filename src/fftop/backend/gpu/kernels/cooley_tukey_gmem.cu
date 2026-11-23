// Cooley-Tukey: in-place DIT. Digit-reverse, then one global-memory kernel per stage.

#include "fftop/backend/gpu/kernels/common.cuh"

namespace FFTop::GPU {
namespace {

__global__ void k_digit_reverse(double2* data, unsigned n, unsigned radix) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) 
        return;
    const unsigned j = static_cast<unsigned>(Math::digit_reverse(i, n, radix));
    if (i < j) {
        double2 tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

// Each thread do a 2-point butterfly
__global__ void k_radix2_stage(double2* data, unsigned n, unsigned stride, unsigned log_stride, const double2* Ws, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) 
        return;
    const unsigned pos = tid & (stride - 1); // position in a group
    const unsigned i0 = (tid >> log_stride) * (stride << 1) + pos; // = no. preceeding groups + pos in group
    const unsigned i1 = i0 + stride;
    const double2 a = data[i0];
    const double2 wb = cmul(load_w(Ws, pos, inverse), data[i1]);
    data[i0] = cadd(a, wb);
    data[i1] = csub(a, wb);
}

// One thread per 4-point block. Ws[(m - 1) stride + k] = W_{4 stride}^{m k}.
__global__ void k_radix4_stage(double2* data, unsigned n, unsigned stride, unsigned log_stride,
                               const double2* Ws, int inverse) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 4) 
        return;

    const unsigned pos = tid & (stride - 1);  // pos in group
    const unsigned i0 = (tid >> log_stride) * (stride << 2) + pos;
    const unsigned i1 = i0 + stride;
    const unsigned i2 = i1 + stride;
    const unsigned i3 = i2 + stride;

    const double2 a0 = data[i0];
    const double2 a1 = cmul(load_w(Ws, pos, inverse), data[i1]);
    const double2 a2 = cmul(load_w(Ws, stride + pos, inverse), data[i2]);
    const double2 a3 = cmul(load_w(Ws, 2 * stride + pos, inverse), data[i3]);

    const double2 even_sum = cadd(a0, a2);
    const double2 even_diff = csub(a0, a2);
    const double2 odd_sum = cadd(a1, a3);
    const double2 odd_diff = csub(a1, a3);
    const double2 odd_turn = inverse ? mul_j(odd_diff) : mul_minus_j(odd_diff);  // turn if inv

    data[i0] = cadd(even_sum, odd_sum);
    data[i1] = cadd(even_diff, odd_turn);
    data[i2] = csub(even_sum, odd_sum);
    data[i3] = csub(even_diff, odd_turn);
}
}  // namespace

Complex* cooley_tukey_gmem(Complex* a, const Complex* W, unsigned n, unsigned radix, Direction dir, unsigned block) {
    double2* data = as_d2(a);
    const int inv = is_inverse(dir);

    // DIT: digit-reverse output
    k_digit_reverse<<<fill_grid(n, block), block>>>(data, n, radix);
    check_cuda(cudaGetLastError(), "digit_reverse");

    auto* stage = radix == 4 ? k_radix4_stage : k_radix2_stage;
    for (unsigned s = 1; s < n; s *= radix) {
        // s < 32 are not coalesced
        stage<<<fill_grid(n / radix, block), block>>>(data, n, s, Math::log2_floor(s), as_d2(W) + (s - 1), inv);
        check_cuda(cudaGetLastError(), "cooley_tukey_gmem_stage");
    }
    return a;
}
}  // namespace FFTop::GPU
