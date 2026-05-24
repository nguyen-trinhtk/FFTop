// Cooley-Tukey's original radix-2 FFT algorithm implementation

#include "fft/core/types.h"

using namespace FFTCore;

void radix_2_fft(const std::vector<FFTCore::Complex>& input, std::vector<FFTCore::Complex>& output) {
    const std::size_t N = input.size();
    output.resize(N);
    if (N <= 1) {
        output = input;
        return;
    }

    // Divide
    std::vector<FFTCore::Complex> even(N / 2);
    std::vector<FFTCore::Complex> odd(N / 2);
    for (std::size_t i = 0; i < N / 2; ++i) {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    // Conquer
    std::vector<Complex> even_fft, odd_fft;
    radix_2_fft(even, even_fft);
    radix_2_fft(odd, odd_fft);

    // Combine
    for (std::size_t k = 0; k < N / 2; ++k) {
        Real theta = (2.0 * PI * k) / N;
        Complex w(std::cos(theta), -std::sin(theta));
        FFTCore::Complex t = w * odd_fft[k];
        output[k] = even_fft[k] + t;
        output[k + N / 2] = even_fft[k] - t;
    }
}