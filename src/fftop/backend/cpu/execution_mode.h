#pragma once

#include <cstddef>
#include <functional>

namespace FFTop::CPU {

inline constexpr std::size_t kParallelGrain = 1024;

class IExecutionMode {
public:
    virtual ~IExecutionMode() = default;

    // Skip the thread fork when n * work_per_item is below kParallelGrain.
    virtual void parallel_for(std::size_t n, std::size_t work_per_item,
                              std::function<void(std::size_t)> fn) const = 0;
};

class SerialExecutionMode final : public IExecutionMode {
public:
    void parallel_for(std::size_t n, std::size_t /*work_per_item*/,
                      std::function<void(std::size_t)> fn) const override {
        for (std::size_t i = 0; i < n; ++i) fn(i);
    }
};

class ParallelExecutionMode final : public IExecutionMode {
public:
    void parallel_for(std::size_t n, std::size_t work_per_item,
                      std::function<void(std::size_t)> fn) const override {
        if (n <= 1 || n * work_per_item < kParallelGrain) {
            for (std::size_t i = 0; i < n; ++i) fn(i);
            return;
        }
#ifdef FFTOP_ENABLE_OPENMP
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) fn(i);
#else
        for (std::size_t i = 0; i < n; ++i) fn(i);
#endif
    }
};

}  // namespace FFTop::CPU
