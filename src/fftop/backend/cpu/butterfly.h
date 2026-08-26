#pragma once

#include "fftop/math/integer.h"
#include "fftop/system.h"
#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

// Combines `radix` interleaved sub-transforms of length `stride` into a single
// transform of length radix * stride, in place. The sub-transform for lane m
// starts at data[offset + m * stride].
class IRadixB {
public:
    virtual ~IRadixB() = default;

    virtual std::size_t radix() const = 0;

    // A single radix only tiles sizes that are a power of that radix; anything
    // else needs mixed radix, which the traversals do not express yet.
    bool supports(std::size_t n) const { return is_power_of(n, radix()); }

    virtual void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                           Direction dir) const = 0;
};

class Radix2 final : public IRadixB {
public:
    explicit Radix2(Simd simd);
    std::size_t radix() const override { return 2; }
    void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                   Direction dir) const override;

private:
    void (*fn_)(Buffer&, std::size_t, std::size_t, Direction);
};

class Radix4 final : public IRadixB {
public:
    explicit Radix4(Simd simd);
    std::size_t radix() const override { return 4; }
    void butterfly(Buffer& data, std::size_t offset, std::size_t stride,
                   Direction dir) const override;

private:
    void (*fn_)(Buffer&, std::size_t, std::size_t, Direction);
};

}  // namespace FFTop::CPU
