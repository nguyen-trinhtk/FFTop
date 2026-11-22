#pragma once

#include "fftop/math/fft_math.h"
#include "fftop/system.h"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

// Generic radix B butterfly network
// For now radix B only support N=B^k
// TODO: generalization (TFT) / mixed radix
class IRadixR {
public:
    virtual ~IRadixR()                = default;
    virtual std::size_t radix() const = 0;
    bool supports(std::size_t n) const { return Math::is_power_of(n, radix()); }

    /// @brief One radix-R butterfly group of contiguous S elements in a stage
    /// @param data pointer to first element
    /// @param stride stride between elements
    /// @param dir forward or inverse
    /// @param Ws pointer to twiddle table of that stage
    virtual void butterfly(Complex* data, std::size_t stride, Direction dir,
                           const Complex* Ws) const = 0;
};

// Radix 2
class Radix2 final : public IRadixR {
public:
    explicit Radix2(SIMD simd);
    std::size_t radix() const override { return 2; }
    void butterfly(Complex* data, std::size_t stride, Direction dir,
                   const Complex* Ws) const override;
private:
    void (*fn_)(Complex*, std::size_t, Direction, const Complex*);
};

// Radix 4
class Radix4 final : public IRadixR {
public:
    explicit Radix4(SIMD simd);
    std::size_t radix() const override { return 4; }
    void butterfly(Complex* data, std::size_t stride, Direction dir,
                   const Complex* Ws) const override;
private:
    void (*fn_)(Complex*, std::size_t, Direction, const Complex*);
};
}  // namespace FFTop::CPU
