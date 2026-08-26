#pragma once

// GPU analogue of cpu/traversal_strategy.h.
// The loop structure is identical to the CPU iterative strategy;
// what changes is that each step is a CUDA kernel launch instead of
// a plain function call.  No Recursive variant — GPU recursion via
// cooperative groups is deferred to the tiled backend.

#include "fftop/types.h"
#include "fftop/backend/gpu/gpu_radix.h"

#include <cstddef>

namespace FFTop::GPU {

class IGPUTraversalStrategy {
public:
    virtual ~IGPUTraversalStrategy() = default;

    // d_data is a device pointer to n Complex elements.
    virtual void run(Complex* d_data, std::size_t n, Direction dir,
                     const IGPURadix& radix) const = 0;
};

// One kernel launch per stage, preceded by a digit-reversal permute kernel.
// Parallel to CPU::IterativeTraversalStrategy.
class IterativeGPUTraversalStrategy final : public IGPUTraversalStrategy {
public:
    void run(Complex* d_data, std::size_t n, Direction dir,
             const IGPURadix& radix) const override;
};

}  // namespace FFTop::GPU
