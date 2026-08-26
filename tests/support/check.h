#pragma once

#include "fftop/types.h"
#include "reference/dft.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>

namespace FFTop::Test {

constexpr Real kTolerance = 1e-12;

constexpr std::size_t kPowersOfTwo[]  = {1, 2, 4, 8, 16, 32, 64};
constexpr std::size_t kPowersOfFour[] = {1, 4, 16, 64, 256};

inline Buffer impulse(std::size_t n) {
    Buffer x(n, {0, 0});
    if (n > 0) x[0] = {1, 0};
    return x;
}

inline Buffer ramp(std::size_t n) {
    Buffer x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = {Real(i), 0};
    return x;
}

// Impulse and ramp are purely real, so their spectra are conjugate-symmetric and
// can hide a real/imaginary mix-up in a SIMD kernel. This one cannot.
inline Buffer mixed(std::size_t n) {
    Buffer x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = {std::cos(Real(i) * Real(0.7)), std::sin(Real(i) * Real(1.3)) + Real(0.25)};
    return x;
}

using Signal = Buffer (*)(std::size_t);
constexpr Signal kSignals[] = {impulse, ramp, mixed};

inline Real relative_error(const Buffer& got, const Buffer& want) {
    if (got.size() != want.size()) return std::numeric_limits<Real>::infinity();
    Real error = 0;
    Real scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        error = std::max(error, std::abs(got[i] - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    return scale > 0 ? error / scale : error;
}

inline void expect_close(const Buffer& got, const Buffer& want, const std::string& label = {}) {
    EXPECT_LT(relative_error(got, want), kTolerance) << label;
}

// FFTop leaves the inverse unnormalised; the reference divides by N.
inline Buffer expected_dft(const Buffer& input, Direction dir) {
    auto want = Ref::dft(input, dir);
    if (dir == Direction::Inverse)
        for (auto& z : want) z *= Real(input.size());
    return want;
}

template <typename Transform>
void expect_matches_dft(Transform&& transform, const Buffer& input, const std::string& name = {}) {
    for (Direction dir : {Direction::Forward, Direction::Inverse}) {
        expect_close(transform(input, dir), expected_dft(input, dir),
                     name + " N=" + std::to_string(input.size()) +
                         (dir == Direction::Forward ? " forward" : " inverse"));
    }
}

template <std::size_t N, typename Fn>
void for_each_input(const std::size_t (&sizes)[N], Fn&& fn) {
    for (auto n : sizes)
        for (auto signal : kSignals) fn(signal(n));
}

}  // namespace FFTop::Test
