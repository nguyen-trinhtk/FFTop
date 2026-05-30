// Cooley-Tukey's original radix-2 FFT algorithm implementation

#include "fft/core/types.h"
#include <cassert>

void radix_2_fft(const std::vector<FFTCore::Complex>& input, std::vector<FFTCore::Complex>& output) {
    const std::size_t N = input.size();
    output.resize(N);
    if (N <= 1) {
        output = input;
        return;
    }

    assert(N >= 2 && (N & (N - 1)) == 0 && (N % 2) == 0);

    // Divide
    std::vector<FFTCore::Complex> even(N / 2);
    std::vector<FFTCore::Complex> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; ++i) {
        // Fill even and odd arrays with input
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    // Conquer
    std::vector<FFTCore::Complex> even_fft, odd_fft;
    radix_2_fft(even, even_fft);
    radix_2_fft(odd, odd_fft);

    // Combine
    for (std::size_t k = 0; k < N / 2; ++k) {
        FFTCore::Real theta = (2.0 * FFTCore::PI * k) / N;
        FFTCore::Complex w(std::cos(theta), -std::sin(theta)); // twiddle
        FFTCore::Complex t = w * odd_fft[k]; 
        // Crossbar between k and k + N/2
        output[k] = even_fft[k] + t;
        output[k + N / 2] = even_fft[k] - t;
    }
}