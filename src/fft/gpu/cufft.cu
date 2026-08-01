#include "fft/gpu/cufft.h"

#include "fft/core/bitops.h"
#include "fft/gpu/detail/cuda_utils.cuh"

#include <cassert>
#include <stdexcept>

#include <cufft.h>

namespace {

struct CufftCache {
    int n = 0;
    cufftHandle plan = 0;
    bool has_plan = false;
    FFTGpu::detail::DeviceBuffer d_in;
    FFTGpu::detail::DeviceBuffer d_out;

    ~CufftCache() {
        reset();
    }

    void reset() {
        if (has_plan) {
            cufftDestroy(plan);
            has_plan = false;
        }
        d_in.reset();
        d_out.reset();
        n = 0;
    }

    void ensure(int size) {
        if (n == size) {
            return;
        }
        reset();
        n = size;
        d_in.allocate(static_cast<std::size_t>(size));
        d_out.allocate(static_cast<std::size_t>(size));
        if (cufftPlan1d(&plan, size, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
            throw std::runtime_error("cufftPlan1d failed");
        }
        has_plan = true;
    }
};

CufftCache& cufft_cache() {
    static thread_local CufftCache cache;
    return cache;
}

}  // namespace

void fft_gpu_cufft(const std::vector<FFTCore::Complex>& input,
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
    static_assert(sizeof(FFTCore::Complex) == sizeof(cufftDoubleComplex),
                  "Complex must match cufftDoubleComplex layout");

    CufftCache& cache = cufft_cache();
    cache.ensure(static_cast<int>(n));
    cache.d_in.upload(input.data());

    if (cufftExecZ2Z(
            cache.plan,
            reinterpret_cast<cufftDoubleComplex*>(cache.d_in.get()),
            reinterpret_cast<cufftDoubleComplex*>(cache.d_out.get()),
            CUFFT_FORWARD) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecZ2Z failed");
    }
    FFT_CUDA_CHECK(cudaDeviceSynchronize());

    cache.d_out.download(output.data());
}
