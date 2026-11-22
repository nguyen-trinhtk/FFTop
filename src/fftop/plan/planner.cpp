#include "fftop/plan/planner.h"

#include "fftop/backend/cpu/cpu_backend.h"
#include "fftop/math/fft_math.h"
#include "fftop/types.h"

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/gpu_backend.h"
#endif

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <variant>

namespace FFTop {

CPUPlanner::CPUPlanner(const SystemConfig& sys, const EstimateParams& params)
    : params_(params)
    , simd_(sys.simd) {
    const std::size_t l2    = sys.l2_bytes ? sys.l2_bytes : params.default_l2;
    const std::size_t cores = sys.cpu_cores ? sys.cpu_cores : sys.cpu_threads;
    const auto budget = static_cast<std::size_t>(static_cast<double>(l2) * params.l2_fraction);
    tile_threshold_   = std::max<std::size_t>(2, Math::floor_pow2(budget / sizeof(Complex)));
    threads_          = sys.openmp ? static_cast<int>(std::max<std::size_t>(1, cores)) : 1;
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

// Tunes block and tile sizes
// Always use smem Stockham, r2, pinned transfer by default
GPUPlanner::GPUPlanner(const GPUInfo& gpu) {
    if (gpu.max_threads_per_block <= 0) return;
    const auto max_threads = static_cast<std::size_t>(gpu.max_threads_per_block);
    cfg_.block_dim = static_cast<unsigned>(std::min<std::size_t>(cfg_.block_dim, max_threads));
    if (gpu.smem_per_block == 0) return;
    // Use largest tile size that fits smem
    // = max(points_in_smem, points_avail)
    const std::size_t fit = std::min(gpu.smem_per_block / (2 * sizeof(Complex)),
                                     max_threads);  // per point uses 2 slots (ping-pong)
    cfg_.tile_size        = static_cast<unsigned>(
        std::max<std::size_t>(2, Math::floor_pow2(fit)));  // cast tile size to a pow2
}

GPUConfig GPUPlanner::plan(std::size_t /*n*/) const { return cfg_; }

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
