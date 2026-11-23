#pragma once

// CUDA-free: configs are built, validated and tested in every build.

#include "fftop/system.h"
#include "fftop/types.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace FFTop {

enum class CPUAlgorithm {
    DFT,        // original O(N^2) (for tiny N)
    Iterative,  // bottom-up
    Recursive,  // top-down
};

// Config for CPU plan
struct CPUConfig {
    CPUAlgorithm algorithm = CPUAlgorithm::Iterative;
    Radix radix = Radix::Radix2;
    SIMD simd = SIMD::Scalar;
    int threads = 1;
    std::size_t split = 0;  // Four-step FFT: N = N1 x N2 with N2 = split
};

enum class GPUKernel { CooleyTukeyGmem, StockhamGmem, StockhamSmem };

// StockhamSmem: columns per block, one 128 B cache line per row (same on every NVIDIA arch)
constexpr unsigned GPU_TILE_WIDTH = 128 / sizeof(Complex);
// StockhamSmem: smem per column point of a block (GPU_TILE_WIDTH points, ping-pong)
constexpr std::size_t GPU_SMEM_ROW_BYTES = 2 * GPU_TILE_WIDTH * sizeof(Complex);

// Config for GPU plan
struct GPUConfig {
    GPUKernel kernel = GPUKernel::StockhamSmem;
    Radix radix = Radix::Radix2;  // Note: Stockham are radix-2 only
    unsigned block_dim = 256;
    unsigned passes = 0;  // StockhamSmem: N = N1 x ... x N_passes, log2 N spread evenly
    bool pinned_staging = true;  // copy through a pinned host buffer
};

// Throw std::invalid_argument if config cannot run size
void validate(const CPUConfig& cfg, std::size_t n);
void validate(const GPUConfig& cfg, std::size_t n, const GPUInfo& gpu);

const char* to_string(CPUAlgorithm algorithm);
const char* to_string(GPUKernel kernel);
std::string to_string(const CPUConfig& cfg);
std::string to_string(const GPUConfig& cfg);

// Specs to run a plan
struct PlanSpec {
    std::size_t size = 0;                       // FFT input size
    std::variant<CPUConfig, GPUConfig> config;  // union
    HardwareTarget target() const {
        // Resolve hardware target
        return std::holds_alternative<CPUConfig>(config) ? HardwareTarget::CPU
                                                         : HardwareTarget::GPU;
    }
};

// Describe a plan spec as string
std::string describe(const PlanSpec& spec);

// A ready-to-run FFT for a (size, HW config) set
// used for both forward and inverse FFT
class Plan {  // abstract class
public:
    virtual ~Plan() = default;
    // The config it was built from, e.g. "CPU/iterative r4 neon t1"
    virtual std::string name() const = 0;  // config
    virtual std::size_t size() const = 0;

    // Execute FFT on host buffers
    // (for GPU, copy from host to device)
    virtual void execute(Buffer& output, const Buffer& input, Direction dir) const = 0;

    // Execute GPU FFT on data already on device
    // in-place is allowed (d_in == d_out)
    virtual void execute_device(Complex* /*d_out*/, const Complex* /*d_in*/,
                                Direction /*dir*/) const {
        throw std::logic_error(name() + " has no device-pointer path");
    }
};

// Build plan from spec
std::unique_ptr<Plan> build_plan(const PlanSpec& spec);
}  // namespace FFTop
