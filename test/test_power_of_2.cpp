#include <vector>
#include "utils.h"

namespace {

FFTTest::TestResult test_power_of_2_sizes(const FFTCore::FFTFunc& fft) {
    const std::vector<int> exponents = {0, 2, 4, 6, 8, 10};

    for (const int exponent : exponents) {
        const int size = 1 << exponent;
        const auto input = FFTTest::generate_random_complex(size);
        std::vector<FFTCore::Complex> expected;
        std::vector<FFTCore::Complex> actual;
        dft(input, expected);
        fft(input, actual);

        const std::string mismatch = FFTTest::find_first_mismatch(expected, actual);
        if (mismatch.empty()) {
            continue;
        }

        return FFTTest::TestResult::fail(
            "N=2^" + std::to_string(exponent) +
            " (" + std::to_string(size) + " samples): " + mismatch);
    }

    return FFTTest::TestResult::pass();
}

const FFTTest::TestRegistrar kPowerOf2Sizes("Power-of-2 sizes", test_power_of_2_sizes);

}  // namespace
