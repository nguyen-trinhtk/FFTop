#pragma once

#include "fftop/backend/ibackend.h"
#include "fftop/backend/gpu/gpu_radix.h"
#include "fftop/backend/gpu/gpu_traversal.h"

#include <cstdint>
#include <memory>
#include <string>

namespace FFTop {

// Abstract base for GPU backends. Owns the is_available() check so
// concrete subclasses don't repeat it.
class GPUBackend : public IBackend {
public:
    bool is_available() const override;
};

// Composed GPU backend: radix + traversal, parallel to CPUBackend.
class GPUFFTBackend final : public GPUBackend {
public:
    GPUFFTBackend(std::unique_ptr<GPU::IGPURadix>             radix,
                  std::unique_ptr<GPU::IGPUTraversalStrategy> traversal,
                  std::string                                 name);

    std::string name() const override { return name_; }
    void execute(const FFTPlan& plan, const Buffer& input, Buffer& output) override;

    // Device-resident path used by the GPU sweep. d_a holds the input of length
    // plan.size; d_b is scratch (may be null if the strategy does not need it);
    // d_W is the length-n twiddle table. Returns the device pointer to the result.
    Complex* execute_device(const FFTPlan& plan, Complex* d_a, Complex* d_b,
                            const Complex* d_W) const;

    bool          needs_scratch() const { return traversal_->needs_scratch(); }
    std::uint64_t estimated_global_bytes(std::size_t n) const {
        return traversal_->estimated_global_bytes(n, radix_->radix());
    }

private:
    std::unique_ptr<GPU::IGPURadix>             radix_;
    std::unique_ptr<GPU::IGPUTraversalStrategy> traversal_;
    std::string                                 name_;
};

}  // namespace FFTop
