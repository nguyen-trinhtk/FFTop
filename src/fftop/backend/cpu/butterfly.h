#pragma once

#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU {

class IRadixB {
public:
    virtual ~IRadixB() = default;
    virtual int  radix() const = 0;
    virtual void butterfly(Buffer& data, std::size_t n,
                           std::size_t offset, std::size_t stride,
                           Direction dir) const = 0;
};

class Radix2 final : public IRadixB {
public:
    int  radix() const override { return 2; }
    void butterfly(Buffer& data, std::size_t n,
                   std::size_t offset, std::size_t stride,
                   Direction dir) const override;
};

class Radix4 final : public IRadixB {
public:
    int  radix() const override { return 4; }
    void butterfly(Buffer& data, std::size_t n,
                   std::size_t offset, std::size_t stride,
                   Direction dir) const override;
};

}  // namespace FFTop::CPU
