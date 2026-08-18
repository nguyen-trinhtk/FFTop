#pragma once

#include "fftop/types.h"

namespace FFTop::CPU::ISA {
using Pack = Complex;

inline Pack load(const Complex& z) { return z; }
inline void store(Complex& z, Pack v) { z = v; }
inline Pack setc(double re, double im) { return {re, im}; }
inline Pack add(Pack a, Pack b) { return a + b; }
inline Pack sub(Pack a, Pack b) { return a - b; }
inline Pack cmul(Pack a, Pack b) { return a * b; }
inline Pack mul_j(Pack z) { return {-z.imag(), z.real()}; }

}  // namespace FFTop::CPU::ISA
