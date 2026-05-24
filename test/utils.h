#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <fft/core/types.h>
#include <fft/ref/dft.h>

namespace FFTTest {

inline constexpr double kTolerance = 1e-10;

struct TestResult {
    bool passed;
    std::string message;

    static TestResult pass();
    static TestResult fail(std::string message);
};

using TestFunction = std::function<TestResult(const FFTCore::FFTFunc&)>;

struct TestCase {
    std::string name;
    TestFunction func;
};

struct Implementation {
    std::string name;
    FFTCore::FFTFunc func;
};

class TestRegistrar {
public:
    TestRegistrar(std::string name, TestFunction func);
};

class ImplementationRegistrar {
public:
    ImplementationRegistrar(std::string name, FFTCore::FFTFunc func);
};

const std::vector<TestCase>& registered_tests();
const std::vector<Implementation>& registered_implementations();
int run_all_tests();

std::vector<FFTCore::Complex> generate_random_complex(std::size_t size);
std::string find_first_mismatch(
    const std::vector<FFTCore::Complex>& expected,
    const std::vector<FFTCore::Complex>& actual,
    double tolerance = kTolerance);
double relative_error(double lhs, double rhs);

}  // namespace FFTTest
