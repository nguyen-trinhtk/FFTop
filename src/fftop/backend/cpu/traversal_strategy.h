#pragma once

#include "fftop/types.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/execution_mode.h"

#include <cstddef>

namespace FFTop::CPU {

class ITraversalStrategy {
public:
    virtual ~ITraversalStrategy() = default;
    virtual void run(Buffer& data, std::size_t n, Direction dir,
                     const IRadixB&         butterfly,
                     const IExecutionMode& execution_mode) const = 0;
};

class RecursiveTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB&         butterfly,
             const IExecutionMode& execution_mode) const override;
private:
    void recurse(Buffer& data, std::size_t n, std::size_t offset,
                 Direction dir, const IRadixB& butterfly,
                 const IExecutionMode& execution_mode) const;
};

class IterativeTraversalStrategy final : public ITraversalStrategy {
public:
    void run(Buffer& data, std::size_t n, Direction dir,
             const IRadixB&         butterfly,
             const IExecutionMode& execution_mode) const override;
};

}  // namespace FFTop::CPU
