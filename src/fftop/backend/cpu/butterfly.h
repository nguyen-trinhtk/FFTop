#pragma once

#include "fftop/math/fft_math.h"
#include "fftop/system.h"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

// Generic radix B butterfly network
// For now radix B only support N=B^k
// TODO: generalization / mixed radix
class IRadixB {
public:
    virtual ~IRadixB() = default;
    virtual std::size_t radix() const = 0;
    bool supports(std::size_t n) const { return Math::is_power_of(n, radix()); } // if N is supported y such matrix
    virtual void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                           Direction dir, const Complex* W, std::size_t n) const = 0;
};

// Radix 2
class Radix2 final : public IRadixB {
public:
    explicit Radix2(SIMD simd);
    std::size_t radix() const override { return 2; }
    void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                   Direction dir, const Complex* W, std::size_t n) const override;
private:
    void (*fn_)(Buffer&, std::size_t, std::size_t, Direction, const Complex*, std::size_t);
};

// Radix 4
class Radix4 final : public IRadixB {
public:
    explicit Radix4(SIMD simd);
    std::size_t radix() const override { return 4; }
    void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                   Direction dir, const Complex* W, std::size_t n) const override;
private:
    void (*fn_)(Buffer&, std::size_t, std::size_t, Direction, const Complex*, std::size_t);
};
}  // namespace FFTop::CPU
