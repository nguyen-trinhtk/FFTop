#include "fftop/plan/plan.h"

#include "fftop/math/fft_math.h"

#include <stdexcept>
#include <string>

namespace FFTop {
namespace {
[[noreturn]] void reject(const CPUConfig& cfg, std::size_t n, const std::string& why) {
    throw std::invalid_argument("CPUConfig {" + to_string(cfg) + "} for N=" + std::to_string(n) +
                                ": " + why);
}

[[noreturn]] void reject(const GPUConfig& cfg, std::size_t n, const std::string& why) {
    throw std::invalid_argument("GPUConfig {" + to_string(cfg) + "} for N=" + std::to_string(n) +
                                ": " + why);
}
constexpr unsigned MAX_THREADS_PER_BLOCK = 1024;
}  // namespace

void validate(const CPUConfig& cfg, std::size_t n) {
    if (cfg.threads < 1) reject(cfg, n, "threads must be >= 1");
    const std::size_t radix = static_cast<std::size_t>(cfg.radix);
    switch (cfg.algorithm) {
        case CPUAlgorithm::DFT:
            if (cfg.split != 0) reject(cfg, n, "DFT can't be split");
            return;
        case CPUAlgorithm::Iterative:
        case CPUAlgorithm::Recursive:
            if (cfg.split == 0) {
                if (n > 1 && !Math::is_power_of(n, radix))
                    reject(cfg, n, "N must be a power of radix " + std::to_string(radix));
                return;
            }
            if (!Math::is_power_of(n, 2)) reject(cfg, n, "N must be a power of 2");
            if (cfg.split < 2 || !Math::is_power_of(cfg.split, 2) || cfg.split >= n)
                reject(cfg, n, "split must be 0 or a power of 2 with 2 <= split < N");
            return;
    }
    reject(cfg, n, "unknown algorithm");
}

const char* to_string(CPUAlgorithm algorithm) {
    switch (algorithm) {
        case CPUAlgorithm::DFT:
            return "dft";
        case CPUAlgorithm::Iterative:
            return "iterative";
        case CPUAlgorithm::Recursive:
            return "recursive";
    }
    return "unknown";
}

std::string to_string(const CPUConfig& cfg) {
    std::string s = to_string(cfg.algorithm);
    if (cfg.algorithm == CPUAlgorithm::DFT) return s;
    if (cfg.split != 0) s += " split=" + std::to_string(cfg.split);
    s += " r" + std::to_string(static_cast<std::size_t>(cfg.radix));
    s += std::string(" ") + to_string(cfg.simd);
    s += " t" + std::to_string(cfg.threads);
    return s;
}

void validate(const GPUConfig& cfg, std::size_t n, const GPUInfo& gpu) {
    if (n > 1 && !Math::is_power_of(n, 2)) reject(cfg, n, "N must be a power of 2");
    if (n > (std::size_t(1) << 31)) reject(cfg, n, "N must fit the kernels' 32-bit indices");

    const unsigned max_threads = gpu.max_threads_per_block > 0
                                     ? static_cast<unsigned>(gpu.max_threads_per_block)
                                     : MAX_THREADS_PER_BLOCK;
    if (cfg.block_dim < 32 || !Math::is_power_of(cfg.block_dim, 2) || cfg.block_dim > max_threads)
        reject(cfg, n, "block must be a power of 2 in [32, " + std::to_string(max_threads) + "]");

    switch (cfg.kernel) {
        case GPUKernel::CooleyTukeyGlobal:
            if (n > 1 && !Math::is_power_of(n, static_cast<std::size_t>(cfg.radix)))
                reject(cfg, n, "N must be a power of the radix");
            return;
        case GPUKernel::StockhamGlobal:
            if (cfg.radix != Radix::Radix2) reject(cfg, n, "Stockham kernels are radix-2 only");
            return;
        case GPUKernel::StockhamShared:
            if (cfg.radix != Radix::Radix2) reject(cfg, n, "Stockham kernels are radix-2 only");
            // One thread per point, two double2 buffers per point in shared memory
            if (cfg.tile_size < 2 || !Math::is_power_of(cfg.tile_size, 2) ||
                cfg.tile_size > max_threads)
                reject(cfg, n,
                       "tile must be a power of 2 in [2, " + std::to_string(max_threads) + "]");
            if (gpu.smem_per_block > 0 && 2 * std::size_t(cfg.tile_size) * 16 > gpu.smem_per_block)
                reject(cfg, n, "tile needs more shared memory than the device has per block");
            return;
    }
    reject(cfg, n, "unknown kernel");
}

const char* to_string(GPUKernel kernel) {
    switch (kernel) {
        case GPUKernel::CooleyTukeyGlobal:
            return "cooley-tukey";
        case GPUKernel::StockhamGlobal:
            return "stockham-global";
        case GPUKernel::StockhamShared:
            return "stockham-shared";
    }
    return "unknown";
}

std::string to_string(const GPUConfig& cfg) {
    std::string s = to_string(cfg.kernel);
    if (cfg.kernel == GPUKernel::StockhamShared) s += " tile=" + std::to_string(cfg.tile_size);
    s += " block=" + std::to_string(cfg.block_dim);
    s += " r" + std::to_string(static_cast<std::size_t>(cfg.radix));
    if (!cfg.pinned_staging) s += " pageable";
    return s;
}

std::string describe(const PlanSpec& spec) {
    const std::string config =
        std::visit([](const auto& cfg) { return to_string(cfg); }, spec.config);
    const char* target = (spec.target() == HardwareTarget::GPU) ? "GPU " : "CPU ";
    return target + config + " N=" + std::to_string(spec.size);
}
}  // namespace FFTop
