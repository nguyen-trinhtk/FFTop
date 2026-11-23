#pragma once

// Host-side GPU backend: plain C++ plus cuda_runtime.h, so .cpp files can include it.
// GPUPlan lives in gpu.cpp, the kernels in kernels/ (one file each).

#include "fftop/plan/plan.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace FFTop {

namespace GPU {
inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

template <bool Pinned>
struct CudaBuffer {
    Complex* ptr = nullptr;

    explicit CudaBuffer(std::size_t n) {
        if (n == 0) return;
        const std::size_t bytes = n * sizeof(Complex);
        check_cuda(Pinned ? cudaMallocHost(&ptr, bytes) : cudaMalloc(&ptr, bytes),
                   Pinned ? "cudaMallocHost" : "cudaMalloc");
    }
    ~CudaBuffer() {
        if (ptr) Pinned ? cudaFreeHost(ptr) : cudaFree(ptr);
    }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
};
using DeviceBuffer = CudaBuffer<false>;
using PinnedBuffer = CudaBuffer<true>;

// TODO: launch GPU DFT for tiny N

/// @brief Launcher for in-place Cooley-Tukey kernel
/// @param a device-pointer to input
/// @param W per-stage twiddle table
/// @param n FFT size
/// @param radix 2 or 4
/// @param dir forward or inverse
/// @param block number of threads per block
/// @return device pointer to a (modified)
Complex* cooley_tukey_gmem(Complex* a, const Complex* W, unsigned n, unsigned radix, Direction dir,
                           unsigned block);

/// @brief Launcher for in-place Stockham (gmem) kernel
/// @param a device-pointer to input
/// @param b device-pointer to scratch buffer (ping-pong)
/// @param W per-stage twiddle table
/// @param n FFT size
/// @param dir forward or inverse
/// @param block number of threads per block
/// @return device pointer to output (either `a` or `b `depending on num stages)
Complex* stockham_gmem(Complex* a, Complex* b, const Complex* W, unsigned n, Direction dir,
                       unsigned block);

/// @brief Launcher for Stockham (smem) kernel: N = N1 x ... x N_passes (log2 n spread evenly),
///        one kernel per factor, each running its column FFTs in shared memory
/// @param a device-pointer to input
/// @param b device-pointer to scratch buffer (ping-pong)
/// @param W per-stage twiddle table
/// @param n FFT size (>= 2)
/// @param dir forward or inverse
/// @param passes number of factors, in [1, log2 n]
/// @param block max threads per block
/// @return device pointer to output (either `a` or `b` depending on num passes)
Complex* stockham_smem(Complex* a, Complex* b, const Complex* W, unsigned n, Direction dir,
                       unsigned passes, unsigned block);

/// @brief Let the Stockham (smem) kernel use up to `bytes` of dynamic smem per block (CUDA caps
///        it at 48 KiB unless the kernel opts in)
/// @note The cap is per kernel, not per plan: pass the device's opt-in limit, so a smaller plan
///       can't lower it under a bigger one
void stockham_smem_opt_in(std::size_t bytes);
}  // namespace GPU

// GPU plan for (size)
// Keeps its device buffers (and a pinned staging buffer) between calls.
class GPUPlan final : public Plan {
public:
    GPUPlan(std::size_t n, const GPUConfig& cfg);

    std::string name() const override { return "GPU/" + to_string(cfg_); }
    std::size_t size() const override { return n_; }

    // Host buffers: H2D, kernels, D2H. Blocks until the result is in output.

    /// @brief Execute kernel on host buffer, with H2D and D2H before/after
    /// @param output pointer to output buffer (host)
    /// @param input pointer to input buffer (host)
    /// @param dir forward or inverse
    void execute(Buffer& output, const Buffer& input, Direction dir) const override;

    /// @brief Execute kernel on device buffer (async on default stream)
    /// @param d_out pointer to output buffer (dev)
    /// @param d_in pointer to input buffer (host)
    /// @param dir forward or inverse
    /// @note Set `d_out = d_in` for in-place execution
    void execute_device(Complex* d_out, const Complex* d_in, Direction dir) const override;

    /// @brief Run kernels only
    /// @param d_a ptr to input buffer (overwritten)
    /// @param d_b ptr to scratch buffer
    /// @param dir forward or inverse
    /// @return ptr to output buffer (either `d_a` or `d_b` depending on kernel & size)
    /// @note Does not include locking/synchronization for callers of different threads
    Complex* run_kernel(Complex* d_a, Complex* d_b, Direction dir) const;

    std::uint64_t estimated_global_bytes() const;  // for calculating bandwidth

private:
    std::size_t n_;
    GPUConfig cfg_;
    std::shared_ptr<const GPU::DeviceBuffer> twiddles_;  // per-stage table, shared across plans

    // Device buffers, reused by calls
    // so calls are serialized (TODO: multithread callers safety)
    mutable std::mutex mu_;
    GPU::DeviceBuffer d_a_;
    GPU::DeviceBuffer d_b_;      // Stockham ping-pong scratch
    GPU::PinnedBuffer staging_;  // when cfg.pinned_staging
};

// Builds the GPU plan for an N-point FFT described by cfg.
// Throws std::runtime_error without a CUDA device, std::invalid_argument
// when cfg can't run N on this device (see validate()).
std::unique_ptr<Plan> make_gpu_plan(std::size_t n, const GPUConfig& cfg);
}  // namespace FFTop
