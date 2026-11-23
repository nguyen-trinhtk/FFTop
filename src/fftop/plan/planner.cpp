#include "fftop/plan/planner.h"

#include "fftop/backend/cpu/cpu.h"
#include "fftop/math/fft_math.h"
#include "fftop/types.h"

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/gpu.h"
#endif

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <variant>

namespace FFTop {
namespace {
constexpr std::size_t DEFAULT_COLUMN = 128;  // 32 KB block (fits default 48KB)
constexpr std::size_t MIN_BLOCKS_PER_SM = 2;  // block sync overlapping

// Largest column-size L for Stockham
// that ensures 
// 1. tile fits smem (opted-in if possible?)
// 2. threads per block under MAX_THREAD_PER_BLOCK
// 3. >= MIN_BLOCKS_PER_SM
std::size_t smem_column(const GPUInfo& gpu) {
    std::size_t column = gpu.smem_per_block_optin > 0
                             ? Math::floor_pow2(gpu.smem_per_block_optin / GPU_SMEM_ROW_BYTES)
                             : DEFAULT_COLUMN;
    const auto fits = [&](std::size_t c) {
        const std::size_t threads = GPU_TILE_WIDTH * c / 2;
        const std::size_t smem = GPU_SMEM_ROW_BYTES * c + gpu.reserved_smem_per_block;
        const auto max_threads = static_cast<std::size_t>(gpu.max_threads_per_block);
        const auto sm_threads = static_cast<std::size_t>(gpu.max_threads_per_sm);
        return (max_threads == 0 || threads <= max_threads) &&
               (sm_threads == 0 || sm_threads / threads >= MIN_BLOCKS_PER_SM) &&
               (gpu.smem_per_sm == 0 || gpu.smem_per_sm / smem >= MIN_BLOCKS_PER_SM);
    };
    while (column > 2 && !fits(column)) column /= 2;
    return column;
}
}  // namespace

CPUPlanner::CPUPlanner(const SystemConfig& sys, const EstimateParams& params)
    : params_(params)
    , simd_(sys.simd) {
    const std::size_t l2 = sys.l2_bytes ? sys.l2_bytes : params.default_l2;
    const std::size_t cores = sys.cpu_cores ? sys.cpu_cores : sys.cpu_threads;
    const auto budget = static_cast<std::size_t>(static_cast<double>(l2) * params.l2_fraction);
    tile_threshold_ = std::max<std::size_t>(2, Math::floor_pow2(budget / sizeof(Complex)));
    threads_ = sys.openmp ? static_cast<int>(std::max<std::size_t>(1, cores)) : 1;
}

std::size_t CPUPlanner::split_for(std::size_t n) const {
    return std::min(tile_threshold_, std::size_t(1) << ((Math::log2_floor(n) + 1) / 2));
}

CPUConfig CPUPlanner::plan(std::size_t n) const {
    CPUConfig cfg;
    cfg.simd = simd_;
    if (n <= params_.dft_max) {
        cfg.algorithm = CPUAlgorithm::DFT;
        return cfg;
    }

    cfg.radix = Math::is_power_of(n, 4) ? Radix::Radix4 : Radix::Radix2;
    // Only parallelize if WPC > threshold
    const std::size_t work = n * Math::log2_floor(n);
    if (work >= params_.parallel_work_per_core * static_cast<std::size_t>(threads_))
        cfg.threads = threads_;
    cfg.algorithm = CPUAlgorithm::Iterative;
    if (n > tile_threshold_) cfg.split = split_for(n);  // four-step
    return cfg;
}

// For now always use smem Stockham, radix-2, pinned transfer by default
GPUPlanner::GPUPlanner(const GPUInfo& gpu)
    : max_column_(smem_column(gpu)) {
    // Block = one thread per butterfly of the longest column
    cfg_.block_dim = static_cast<unsigned>(std::max<std::size_t>(32, GPU_TILE_WIDTH * max_column_ / 2));
}

// Fewest passes with every column <= L, spreaded evenly
GPUConfig GPUPlanner::plan(std::size_t n) const {
    GPUConfig cfg = cfg_;
    cfg.passes = static_cast<unsigned>(
        Math::ceil_div(Math::log2_floor(n), Math::log2_floor(max_column_)));
    return cfg;
}

Planner::Planner(SystemConfig system, EstimateParams params, PlanCache* cache)
    : system_(std::move(system))
    , params_(params)
    , cache_(cache) {}

HardwareTarget Planner::resolve(std::size_t size, HardwareTarget target) const {
    switch (target) {
        case HardwareTarget::CPU:
            return HardwareTarget::CPU;
        case HardwareTarget::GPU:
            if (!system_.cuda)
                throw std::runtime_error("GPU backend requested but no CUDA device is available");
            return HardwareTarget::GPU;
        case HardwareTarget::Auto:
            // Run on CPU for smaller FFTs since H2D/D2H transfer cost might dominates
            return system_.cuda && size >= params_.gpu_min ? HardwareTarget::GPU
                                                           : HardwareTarget::CPU;
    }
    return HardwareTarget::CPU;
}

PlanSpec Planner::make_spec(std::size_t size, HardwareTarget target) const {
    // TODO: truncated FFT or mixed radix for other sizes
    assert((size == 0 || Math::is_power_of(size, 2)) && "N must be a power of 2");
    PlanSpec spec;
    spec.size = size;
    if (resolve(size, target) == HardwareTarget::GPU)
        spec.config = GPUPlanner(system_.gpu).plan(size);
    else
        spec.config = CPUPlanner(system_, params_).plan(size);
    return spec;
}

std::shared_ptr<const Plan> Planner::plan(std::size_t size, HardwareTarget target) const {
    // Find plan in cache
    if (cache_) {
        if (auto hit = cache_->find(size, target)) return hit;
    }
    std::shared_ptr<const Plan> built = build_plan(make_spec(size, target));
    if (cache_) return cache_->store(size, target, std::move(built));
    return built;
}

std::unique_ptr<Plan> build_plan(const PlanSpec& spec) {
    if (const auto* cpu = std::get_if<CPUConfig>(&spec.config))
        return make_cpu_plan(spec.size, *cpu);
#if defined(FFTOP_ENABLE_CUDA)
    return make_gpu_plan(spec.size, std::get<GPUConfig>(spec.config));
#else
    throw std::runtime_error("GPU backend requested but FFTop was built without CUDA");
#endif
}
}  // namespace FFTop
