#pragma once

#include "fftop/types.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/execution_mode.h"

#include <cstddef>

namespace FFTop::CPU {
// Abstract interface
class ITraversalStrategy {
public:
    virtual ~ITraversalStrategy() = default;
    virtual void run(Buffer& data, std::size_t n, Direction dir,
                     const IRadixB& butterfly,
                     const IExecutionMode& execution_mode,
                     const Complex* W) const = 0;
};

// Recursive traversal
class RecursiveTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB& butterfly,
             const IExecutionMode& execution_mode,
             const Complex* W) const override;
private:
    void recurse(Buffer& data, std::size_t n, std::size_t offset,
                 Direction dir, const IRadixB& butterfly,
                 const IExecutionMode& execution_mode,
                 const Complex* W, std::size_t n_full) const;
};

// Iterative traversal
class IterativeTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB& butterfly,
             const IExecutionMode& execution_mode,
             const Complex* W) const override;
};
}  // namespace FFTop::CPU
