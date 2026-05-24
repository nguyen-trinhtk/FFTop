#include <cmath>
#include <vector>
#include "utils.h"

namespace {

FFTTest::TestResult test_parseval_theorem(const FFTCore::FFTFunc& fft) {
    const std::vector<int> exponents = {2, 3, 4, 5, 6, 8, 10, 12};

    for (const int exponent : exponents) {
        const int size = 1 << exponent;
        const auto input = FFTTest::generate_random_complex(size);

        double time_energy = 0.0;
        for (const auto& x : input) {
            time_energy += std::norm(x);
        }

        std::vector<FFTCore::Complex> freq;
        fft(input, freq);
        double freq_energy = 0.0;
        for (const auto& X : freq) {
            freq_energy += std::norm(X);
        }

        const double normalized_freq_energy = freq_energy / size;
        const double error = FFTTest::relative_error(time_energy, normalized_freq_energy);

        if (error > FFTTest::kTolerance * 10) {
            return FFTTest::TestResult::fail(
                "N=2^" + std::to_string(exponent) +
                " (" + std::to_string(size) + " samples): relative error=" +
                std::to_string(error));
        }
    }

    return FFTTest::TestResult::pass();
}

const FFTTest::TestRegistrar kParseval("Parseval's theorem", test_parseval_theorem);

}  // namespace
