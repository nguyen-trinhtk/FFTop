#pragma once

// GPU analogue of cpu/traversal_strategy.h.
// Cooley-Tukey is in-place (scratch unused). Stockham ping-pongs a/b.

#include "fftop/types.h"
#include "fftop/backend/gpu/gpu_radix.h"

#include <cstddef>
#include <cstdint>

namespace FFTop::GPU {

class IGPUTraversalStrategy {
public:
    virtual ~IGPUTraversalStrategy() = default;

    // a holds the input. b is scratch (unused by in-place Cooley-Tukey).
    // W is a length-n device table, W[k] = cis(-2π k / n).
    // Returns the device pointer that holds the result (a or b).
    virtual Complex* run(Complex* a, Complex* b, const Complex* W, std::size_t n,
                         Direction dir, const IGPURadix& radix) const = 0;

    virtual bool          needs_scratch() const { return false; }
    virtual std::uint64_t estimated_global_bytes(std::size_t n,
                                                 std::size_t radix) const = 0;
};

// In-place DIT: digit-reversal permute, then one butterfly kernel per stage.
class IterativeGPUTraversalStrategy final : public IGPUTraversalStrategy {
public:
    Complex* run(Complex* a, Complex* b, const Complex* W, std::size_t n, Direction dir,
                 const IGPURadix& radix) const override;
    std::uint64_t estimated_global_bytes(std::size_t n, std::size_t radix) const override;
};

// Out-of-place self-sorting Stockham: one global-memory kernel per radix-2 stage.
// No bit-reversal. Ping-pongs a/b.
class StockhamGlobalTraversalStrategy final : public IGPUTraversalStrategy {
public:
    Complex* run(Complex* a, Complex* b, const Complex* W, std::size_t n, Direction dir,
                 const IGPURadix& radix) const override;
    bool          needs_scratch() const override { return true; }
    std::uint64_t estimated_global_bytes(std::size_t n, std::size_t radix) const override;
};

// Stockham with shared-memory inner FFTs. N that fit in a tile run entirely in
// shared memory. Larger N is factored as N1×N2 (four-step): batched tile-sized
// Stockham in shared memory, twiddle, transpose, recurse.
class StockhamSharedTraversalStrategy final : public IGPUTraversalStrategy {
public:
    Complex* run(Complex* a, Complex* b, const Complex* W, std::size_t n, Direction dir,
                 const IGPURadix& radix) const override;
    bool          needs_scratch() const override { return true; }
    std::uint64_t estimated_global_bytes(std::size_t n, std::size_t radix) const override;
};

}  // namespace FFTop::GPU
