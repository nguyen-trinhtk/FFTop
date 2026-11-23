#include "fftop/backend/gpu/gpu.h"

#include "fftop/math/fft_math.h"
#include "fftop/math/twiddle.h"
#include "fftop/system.h"

#include <cstring>
#include <tuple>
#include <utility>

namespace FFTop {
namespace {

using GPU::check_cuda;

// On-device twiddle table for (n, radix)
std::shared_ptr<const GPU::DeviceBuffer> device_twiddles(std::size_t n, std::size_t radix) {
    using Key = std::tuple<int, std::size_t, std::size_t>;
    static auto* cache = new TableCache<Key, GPU::DeviceBuffer>();

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    return cache->get({device, n, radix}, [&] {
        const Twiddles host = make_twiddles(n, radix);  // freed after upload
        auto d = std::make_shared<GPU::DeviceBuffer>(host.w.size());
        check_cuda(cudaMemcpy(d->ptr, host.w.data(), host.w.size() * sizeof(Complex),
                              cudaMemcpyHostToDevice),"twiddle H2D");
        return std::shared_ptr<const GPU::DeviceBuffer>(std::move(d));
    });
}
}  // namespace

GPUPlan::GPUPlan(std::size_t n, const GPUConfig& cfg)
    : n_(n), cfg_(cfg), twiddles_(n > 1 ? device_twiddles(n, static_cast<std::size_t>(cfg.radix)) : nullptr)
    , d_a_(n), d_b_(cfg.kernel == GPUKernel::CooleyTukeyGmem ? 0 : n), staging_(cfg.pinned_staging ? n : 0) {
    if (cfg.kernel == GPUKernel::StockhamSmem)
        GPU::stockham_smem_opt_in(system_config().gpu.smem_per_block_optin);
}

Complex* GPUPlan::run_kernel(Complex* d_a, Complex* d_b, Direction dir) const {
    if (n_ <= 1) 
        return d_a;
    const Complex* W = twiddles_->ptr;
    const auto n = static_cast<unsigned>(n_);
    switch (cfg_.kernel) {
        case GPUKernel::CooleyTukeyGmem:
            return GPU::cooley_tukey_gmem(d_a, W, n, static_cast<unsigned>(cfg_.radix), dir, cfg_.block_dim);
        case GPUKernel::StockhamGmem:
            return GPU::stockham_gmem(d_a, d_b, W, n, dir, cfg_.block_dim);
        case GPUKernel::StockhamSmem:
            return GPU::stockham_smem(d_a, d_b, W, n, dir, cfg_.passes, cfg_.block_dim);
    }
    return d_a;
}

std::uint64_t GPUPlan::estimated_global_bytes() const {
    if (n_ <= 1) 
        return 0;
    const std::uint64_t pass = 2ull * n_ * sizeof(Complex);  // one read + one write
    switch (cfg_.kernel) {
        case GPUKernel::CooleyTukeyGmem: {
            std::uint64_t stages = 0;
            for (std::size_t s = 1; s < n_; s *= static_cast<std::size_t>(cfg_.radix)) 
                ++stages;
            return (1 + stages) * pass;  // digit-reverse + each stage
        }
        case GPUKernel::StockhamGmem:
            return Math::log2_floor(n_) * pass;
        case GPUKernel::StockhamSmem:
            return cfg_.passes * pass;
    }
    return 0;
}

void GPUPlan::execute(Buffer& output, const Buffer& input, Direction dir) const {
    if (input.size() != n_)
        throw std::invalid_argument(name() + ": built for N=" + std::to_string(n_) + ", got " + std::to_string(input.size()));
    output.resize(n_);
    if (n_ == 0) 
        return;

    // Protected execution
    // TODO: multi streams?
    std::lock_guard<std::mutex> lock(mu_);
    const std::size_t bytes = n_ * sizeof(Complex);

    const Complex* src = input.data();
    if (staging_.ptr) {
        // Pinned staging: one host memcpy then DMA
        // TODO: pin limits?
        std::memcpy(staging_.ptr, input.data(), bytes);
        src = staging_.ptr;
    }

    // H2D
    check_cuda(cudaMemcpy(d_a_.ptr, src, bytes, cudaMemcpyHostToDevice), "GPU H2D");

    // Kernel execution
    const Complex* d_out = run_kernel(d_a_.ptr, d_b_.ptr, dir);
    check_cuda(cudaDeviceSynchronize(), "GPU kernel");
    Complex* dst = staging_.ptr ? staging_.ptr : output.data();

    // D2H
    check_cuda(cudaMemcpy(dst, d_out, bytes, cudaMemcpyDeviceToHost), "GPU D2H");
    if (staging_.ptr) 
        std::memcpy(output.data(), staging_.ptr, bytes);
}

void GPUPlan::execute_device(Complex* d_out, const Complex* d_in, Direction dir) const {
    if (n_ == 0) 
        return;
    std::lock_guard<std::mutex> lock(mu_);
    const std::size_t bytes = n_ * sizeof(Complex);

    // Copy d_in to d_out then work on d_out only
    if (d_out != d_in)
        check_cuda(cudaMemcpyAsync(d_out, d_in, bytes, cudaMemcpyDeviceToDevice, 0), "GPU D2D in");
    const Complex* result = run_kernel(d_out, d_b_.ptr, dir);
    if (result != d_out)
        check_cuda(cudaMemcpyAsync(d_out, result, bytes, cudaMemcpyDeviceToDevice, 0),"GPU D2D out");
}

std::unique_ptr<Plan> make_gpu_plan(std::size_t n, const GPUConfig& cfg) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        throw std::runtime_error("GPU backend requested but no CUDA device is available");
    validate(cfg, n, system_config().gpu);
    return std::make_unique<GPUPlan>(n, cfg);
}
}  // namespace FFTop
