#pragma once

#include "fftop/types.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/execution_mode.h"
#include "fftop/math/twiddle.h"

#include <cstddef>

namespace FFTop::CPU {
// Abstract interface
// T is the per-stage twiddle table for that radix
// where each stage with stride s reads T.stage(s)
class ITraversalStrategy {
public:
    virtual ~ITraversalStrategy() = default;
    virtual void run(Buffer& data, std::size_t n, Direction dir,
                     const IRadixB& butterfly,
                     bool parallel,
                     const Twiddles& T) const = 0;
};

// Recursive traversal
class RecursiveTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB& butterfly,
             bool parallel,
             const Twiddles& T) const override;
private:
    void recurse(Buffer& data, std::size_t n, std::size_t offset,
                 Direction dir, const IRadixB& butterfly,
                 bool parallel,
                 const Twiddles& T) const;
};

// Iterative traversal
class IterativeTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB& butterfly,
             bool parallel,
             const Twiddles& T) const override;
};
}  // namespace FFTop::CPU
