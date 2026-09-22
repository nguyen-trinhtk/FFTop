#include "fftop/backend/gpu/gpu_backend.h"
#include "fftop/backend/gpu/gpu_utils.h"
#include "fftop/math/twiddle.h"

#include <cassert>
#include <cuda_runtime.h>

namespace FFTop {

bool GPUBackend::is_available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

GPUFFTBackend::GPUFFTBackend(std::unique_ptr<GPU::IGPURadix>             radix,
                             std::unique_ptr<GPU::IGPUTraversalStrategy> traversal,
                             std::string                                 name)
    : radix_(std::move(radix))
    , traversal_(std::move(traversal))
    , name_(std::move(name)) {}

Complex* GPUFFTBackend::execute_device(const FFTPlan& plan, Complex* d_a, Complex* d_b,
                                       const Complex* d_W) const {
    if (plan.size <= 1) return d_a;
    return traversal_->run(d_a, d_b, d_W, plan.size, plan.direction, *radix_);
}

void GPUFFTBackend::execute(const FFTPlan& plan, const Buffer& input, Buffer& output) {
    output.resize(plan.size);
    if (plan.size == 0) return;
    assert(input.size() >= plan.size);

    GPU::DeviceBuf d_a(plan.size);
    GPU::DeviceBuf d_b(needs_scratch() ? plan.size : 0);
    GPU::DeviceBuf d_W(plan.size);

    GPU::check_cuda(
        cudaMemcpy(d_a.ptr, input.data(), plan.size * sizeof(double2), cudaMemcpyHostToDevice),
        "GPU H2D");

    const auto W = default_twiddles().get(plan.size);
    GPU::check_cuda(
        cudaMemcpy(d_W.ptr, W->data(), plan.size * sizeof(double2), cudaMemcpyHostToDevice),
        "GPU twiddle H2D");

    Complex* d_out = execute_device(plan, reinterpret_cast<Complex*>(d_a.ptr),
                                    reinterpret_cast<Complex*>(d_b.ptr),
                                    reinterpret_cast<const Complex*>(d_W.ptr));

    GPU::check_cuda(cudaDeviceSynchronize(), "GPU kernel");
    GPU::check_cuda(
        cudaMemcpy(output.data(), d_out, plan.size * sizeof(double2), cudaMemcpyDeviceToHost),
        "GPU D2H");
}

}  // namespace FFTop
