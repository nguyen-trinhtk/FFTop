#pragma once

#include "fftop/plan/cache.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"

#include <cstddef>
#include <memory>

namespace FFTop {
// Thresholds
// TODO: calibrate after bench
// TODO (maybe): measure mode (similar to cuDNN benchmark)
struct EstimateParams {
    std::size_t dft_max = 8;  // run direct DFT for N <= dft_max
    std::size_t parallel_work_per_core = std::size_t(1)
                                         << 15;  // only parallelize if Nlog2(N) >= wpc * n_threads
    double l2_fraction = 0.5;                    // fraction of L2 an in-cache FFT can use
    std::size_t default_l2 = std::size_t(1) << 20;
    std::size_t gpu_min = std::size_t(1) << 16;  // (for Auto only): use GPU only if N >= gpu_min
};

// Plan CPUConfig from hardware specs
class CPUPlanner {
public:
    CPUPlanner(const SystemConfig& sys, const EstimateParams& params);
    CPUConfig plan(std::size_t n) const;
    std::size_t tile_threshold() const {
        return tile_threshold_;
    }                                            // only split if N > tile_threshold
    std::size_t split_for(std::size_t n) const;  // N2 ~= sqrt(N)
private:
    EstimateParams params_;
    SIMD simd_;
    int threads_;
    std::size_t tile_threshold_;  // l2_frac * l2_size
};

// Plan GPUConfig from device specs
class GPUPlanner {
public:
    explicit GPUPlanner(const GPUInfo& gpu);
    GPUConfig plan(std::size_t n) const;
private:
    GPUConfig cfg_;
    std::size_t max_column_;  // longest StockhamSmem column (power of 2)
};

class Planner {
public:
    explicit Planner(SystemConfig system, EstimateParams params = {}, PlanCache* cache = nullptr);

    PlanSpec make_spec(std::size_t size, HardwareTarget target) const;
    std::shared_ptr<const Plan> plan(std::size_t size, HardwareTarget target) const;
private:
    // Resolve auto
    HardwareTarget resolve(std::size_t size, HardwareTarget target) const;
    SystemConfig system_;
    EstimateParams params_;
    PlanCache* cache_;
};
}  // namespace FFTop
