#pragma once

#include "fftop/types.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/parallel.h"
#include "fftop/math/twiddle.h"

#include <cstddef>

namespace FFTop::CPU {
class ITraversalStrategy {
public:
    virtual ~ITraversalStrategy() = default;

    /// @brief In-place FFT on N contiguous elements
    /// @param data Data buffer
    /// @param n data size
    /// @param dir Forward or inverse
    /// @param butterfly radix-R butterfly
    /// @param threads number of OMP threads (serial `1`)
    /// @param T Per-stage twiddle table, with stride `s` reads `T.stage(s)`
    virtual void run(Complex* data, std::size_t n, Direction dir, const IRadixR& butterfly,
                     int threads, const Twiddles& T) const = 0;
};

// Recursive traversal
class RecursiveTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Complex* data, std::size_t n, Direction dir, const IRadixR& butterfly, int threads,
             const Twiddles& T) const override;
private:
    void recurse(Complex* data, std::size_t n, Direction dir, const IRadixR& butterfly, int threads,
                 const Twiddles& T) const;
};

// Iterative traversal
class IterativeTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Complex* data, std::size_t n, Direction dir, const IRadixR& butterfly, int threads,
             const Twiddles& T) const override;
};
}  // namespace FFTop::CPU
