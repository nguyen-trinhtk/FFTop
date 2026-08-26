#pragma once

// GPU analogue of cpu/butterfly.h.
// IGPURadix owns the CUDA kernel for one butterfly stage.
// No SIMD dispatch — the GPU thread model replaces it.
// No IExecutionMode — the thread grid is always fully parallel.

#include "fftop/types.h"

#include <cstddef>
#include <memory>

namespace FFTop::GPU {

class IGPURadix {
public:
    virtual ~IGPURadix() = default;

    // Launch a CUDA kernel that executes one butterfly stage across all N/radix
    // butterfly groups. d_data is a device pointer; the caller owns it.
    // Synchronisation between stages is handled by the default CUDA stream.
    virtual void butterfly_stage(Complex* d_data, std::size_t n,
                                 std::size_t stride, Direction dir) const = 0;

    virtual std::size_t radix()            const = 0;
    virtual bool        supports(std::size_t n) const = 0;
};

// Global-memory-only radix-2.  Parallel to CPU::Radix2.
class GPURadix2 final : public IGPURadix {
public:
    void        butterfly_stage(Complex* d_data, std::size_t n,
                                std::size_t stride, Direction dir) const override;
    std::size_t radix()            const override { return 2; }
    bool        supports(std::size_t n) const override;
};

// Global-memory-only radix-4.  Parallel to CPU::Radix4.
class GPURadix4 final : public IGPURadix {
public:
    void        butterfly_stage(Complex* d_data, std::size_t n,
                                std::size_t stride, Direction dir) const override;
    std::size_t radix()            const override { return 4; }
    bool        supports(std::size_t n) const override;
};

}  // namespace FFTop::GPU
