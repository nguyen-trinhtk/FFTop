#pragma once

#include <cstddef>

namespace FFTop {

// True when n == base^k for some k >= 0.
inline bool is_power_of(std::size_t n, std::size_t base) {
    if (n == 0 || base < 2) return false;
    while (n % base == 0) n /= base;
    return n == 1;
}

}  // namespace FFTop
