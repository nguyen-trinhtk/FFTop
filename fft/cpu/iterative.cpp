// Iterative in-place radix-2 FFT

#include "fft/core/types.h"
#include "fft/core/bitops.h"
#include <cassert>

using namespace FFTCore;

void fft_iterative(const std::vector<FFTCore::Complex>& input, std::vector<FFTCore::Complex>& output) {
    const std::size_t N = input.size();
    assert(N >= 1 && (N & (N - 1)) == 0);

    output = input;

    bit_reverse_permute(output);

    for (std::size_t len = 2; len <= N; len <<= 1) {
        FFTCore::Real theta = -2.0 * PI / len;
        FFTCore::Complex wlen(std::cos(theta), std::sin(theta));

        for (std::size_t i = 0; i < N; i += len) {
            FFTCore::Complex w(1.0, 0.0);
            for (std::size_t j = 0; j < len / 2; ++j) {
                FFTCore::Complex u = output[i + j];
                FFTCore::Complex t = w * output[i + j + len / 2];
                output[i + j] = u + t;
                output[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}