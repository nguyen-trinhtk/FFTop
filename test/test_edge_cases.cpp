#include <vector>
#include "utils.h"

namespace {

FFTTest::TestResult verify_matches_reference(
    const FFTCore::FFTFunc& fft,
    const std::vector<FFTCore::Complex>& input,
    const std::string& context) {
    std::vector<FFTCore::Complex> expected;
    std::vector<FFTCore::Complex> actual;
    dft(input, expected);
    fft(input, actual);

    const std::string mismatch = FFTTest::find_first_mismatch(expected, actual);
    if (mismatch.empty()) {
        return FFTTest::TestResult::pass();
    }

    return FFTTest::TestResult::fail(context + ": " + mismatch);
}

FFTTest::TestResult test_all_zeros(const FFTCore::FFTFunc& fft) {
    const int size = 1024;
    std::vector<FFTCore::Complex> input(size, {0.0, 0.0});
    return verify_matches_reference(fft, input, "all-zero input");
}

const FFTTest::TestRegistrar kAllZeros("All zeros", test_all_zeros);

FFTTest::TestResult test_impulse_at_zero(const FFTCore::FFTFunc& fft) {
    const int size = 1024;
    std::vector<FFTCore::Complex> input(size, {0.0, 0.0});
    input[0] = {1.0, 0.0};
    return verify_matches_reference(fft, input, "impulse at index 0");
}

const FFTTest::TestRegistrar kImpulseAtZero("Impulse at zero", test_impulse_at_zero);

FFTTest::TestResult test_impulse_at_k(const FFTCore::FFTFunc& fft) {
    const int size = 1024;
    const int impulse_index = 7;

    std::vector<FFTCore::Complex> input(size, {0.0, 0.0});
    input[impulse_index] = {1.0, 0.0};
    return verify_matches_reference(
        fft,
        input,
        "impulse at index " + std::to_string(impulse_index));
}

const FFTTest::TestRegistrar kImpulseAtK("Impulse at k", test_impulse_at_k);

}  // namespace
