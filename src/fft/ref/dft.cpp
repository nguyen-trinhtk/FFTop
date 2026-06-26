#include "fft/ref/dft.h"

void dft(const std::vector<Complex>& input, std::vector<Complex>& output) {
    size_t N = input.size();
    output.resize(N);
    
    for (size_t k = 0; k < N; ++k) {
        output[k] = Complex(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            Real theta = (2.0 * PI * k * n) / N;
            Complex w(std::cos(theta), -std::sin(theta));   
            output[k] += input[n] * w;
        }
    }
}
