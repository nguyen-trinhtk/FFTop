#pragma once

#include "fftop/types.h"

#include <cstddef>

namespace FFTop::CPU::Scalar {

static constexpr std::size_t width = 1;
using Pack = Complex;

inline Pack load(const Complex& z) { return z; }
inline void store(Complex& z, Pack v) { z = v; }
inline Pack load_pack(const Real* p) { return {p[0], p[1]}; }
inline Pack setc(double re, double im) { return {re, im}; }
inline Pack add(Pack a, Pack b) { return a + b; }
inline Pack sub(Pack a, Pack b) { return a - b; }
inline Pack cmul(Pack a, Pack b) { return a * b; }
inline Pack mul_j(Pack z)       { return {-z.imag(), z.real()}; }
inline Pack mul_minus_j(Pack z) { return {z.imag(), -z.real()}; }
inline Pack conj(Pack z)        { return {z.real(), -z.imag()}; }

}  // namespace FFTop::CPU::Scalar
