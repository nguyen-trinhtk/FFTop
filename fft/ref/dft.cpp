#include <iostream> 
#include <vector>
#include <cmath>

#include "fft/core/types.h"

using namespace FFTCore;

std::vector<Complex> dft(const std::vector<Complex>& input) {
    size_t N = input.size();
    std::vector<Complex> output(N);
    const Real PI = std::acos(-1.0);
    
    for (size_t k = 0; k < N; ++k) {
        output[k] = Complex(0.0, 0.0);
        for (size_t n = 0; n < N; ++n) {
            Real theta = (2.0 * PI * k * n) / N;
            Complex w(std::cos(theta), -std::sin(theta));   
            output[k] += input[n] * w;
        }
    }
    return output;
}