// Radix-4 FFT implementation

#include "fft/cpu/radix-4.h"

#include "fft/core/types.h"
#include "fft/core/complex-utils.h"
#include <cassert>

void radix_4_fft(const std::vector<FFTCore::Complex> &input, std::vector<FFTCore::Complex> &output)
{
    const std::size_t N = input.size();
    output.resize(N);

    // Base cases: sizes 1 and 2 can be computed directly.
    if (N <= 1)
    {
        output = input;
        return;
    }

    if (N == 2)
    {
        output[0] = input[0] + input[1];
        output[1] = input[0] - input[1];
        return;
    }

    // N must be a power of 4; caller is responsible for this invariant
    assert((N & (N - 1)) == 0 && (N % 4) == 0);

    // Divide — split by index mod 4
    std::vector<FFTCore::Complex> g0(N / 4), g1(N / 4), g2(N / 4), g3(N / 4);
    for (std::size_t i = 0; i < N / 4; ++i)
    {
        g0[i] = input[4 * i];
        g1[i] = input[4 * i + 1];
        g2[i] = input[4 * i + 2];
        g3[i] = input[4 * i + 3];
    }

    // Conquer
    std::vector<FFTCore::Complex> f0, f1, f2, f3;
    radix_4_fft(g0, f0);
    radix_4_fft(g1, f1);
    radix_4_fft(g2, f2);
    radix_4_fft(g3, f3);

    // Combine — radix-4 butterfly
    for (std::size_t k = 0; k < N / 4; ++k)
    {
        FFTCore::Real theta = (2.0 * FFTCore::PI * k) / N;
        FFTCore::Complex w1(std::cos(theta), -std::sin(theta));         // W^k
        FFTCore::Complex w2(std::cos(2 * theta), -std::sin(2 * theta)); // W^2k
        FFTCore::Complex w3(std::cos(3 * theta), -std::sin(3 * theta)); // W^3k

        // Apply twiddles (W^0 = 1, so t0 is free)
        FFTCore::Complex t0 = f0[k];
        FFTCore::Complex t1 = w1 * f1[k];
        FFTCore::Complex t2 = w2 * f2[k];
        FFTCore::Complex t3 = w3 * f3[k];

        // 4-point DFT core — only ±1 and ±i, no multiplies
        FFTCore::Complex a0 = t0 + t2;
        FFTCore::Complex a1 = t0 - t2;
        FFTCore::Complex a2 = t1 + t3;
        FFTCore::Complex a3 = FFTCore::times_neg_i(t1 - t3); // ×(-i) here, once, on a3 only

        // Butterfly outputs — plain ± from here, no more rotations
        output[k] = a0 + a2;
        output[k + N / 4] = a1 + a3;
        output[k + N / 2] = a0 - a2;
        output[k + 3 * N / 4] = a1 - a3;
    }
}