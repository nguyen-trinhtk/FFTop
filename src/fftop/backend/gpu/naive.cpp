#include "fftop/backend/gpu/naive.h"

#include <cassert>

namespace FFTop {

#if defined(FFTOP_ENABLE_CUDA)
void naive_dft_cuda(const Complex* input, Complex* output, std::size_t n, Direction dir);
#endif

void NaiveGPUBackend::execute(const FFTPlan& plan, const Buffer& input, Buffer& output) {
    output.resize(plan.size);
    if (plan.size == 0) return;
    assert(input.size() >= plan.size);

#if defined(FFTOP_ENABLE_CUDA)
    naive_dft_cuda(input.data(), output.data(), plan.size, plan.direction);
#else
    (void)input;
    assert(false && "NaiveGPUBackend requires FFTOP_ENABLE_CUDA");
#endif
}

}  // namespace FFTop
