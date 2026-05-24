#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>

#include <fft/ref/dft.h>

namespace FFTTest {
namespace {

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> tests;
    return tests;
}

std::vector<Implementation>& implementation_registry() {
    static std::vector<Implementation> implementations;
    return implementations;
}

bool complex_equal(
    const FFTCore::Complex& lhs,
    const FFTCore::Complex& rhs,
    double tolerance) {
    return std::abs(lhs.real() - rhs.real()) < tolerance &&
           std::abs(lhs.imag() - rhs.imag()) < tolerance;
}

}  // namespace

TestResult TestResult::pass() {
    return {true, ""};
}

TestResult TestResult::fail(std::string message) {
    return {false, std::move(message)};
}

TestRegistrar::TestRegistrar(std::string name, TestFunction func) {
    test_registry().push_back({std::move(name), std::move(func)});
}

ImplementationRegistrar::ImplementationRegistrar(std::string name, FFTCore::FFTFunc func) {
    implementation_registry().push_back({std::move(name), std::move(func)});
}

const std::vector<TestCase>& registered_tests() {
    return test_registry();
}

const std::vector<Implementation>& registered_implementations() {
    return implementation_registry();
}

std::vector<FFTCore::Complex> generate_random_complex(std::size_t size) {
    static std::mt19937 generator(1337);
    static std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    std::vector<FFTCore::Complex> data(size);
    for (auto& value : data) {
        value = FFTCore::Complex(distribution(generator), distribution(generator));
    }
    return data;
}

std::string find_first_mismatch(
    const std::vector<FFTCore::Complex>& expected,
    const std::vector<FFTCore::Complex>& actual,
    double tolerance) {
    if (expected.size() != actual.size()) {
        std::ostringstream message;
        message << "size mismatch: expected " << expected.size()
                << " bins but got " << actual.size();
        return message.str();
    }

    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (complex_equal(expected[index], actual[index], tolerance)) {
            continue;
        }

        std::ostringstream message;
        message << std::fixed << std::setprecision(6)
                << "mismatch at bin " << index
                << ": expected (" << expected[index].real() << ", " << expected[index].imag() << ")"
                << ", got (" << actual[index].real() << ", " << actual[index].imag() << ")";
        return message.str();
    }

    return "";
}

double relative_error(double lhs, double rhs) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    if (scale == 0.0) {
        return 0.0;
    }
    return std::abs(lhs - rhs) / scale;
}

int run_all_tests() {
    const std::vector<TestCase>& tests = registered_tests();
    const std::vector<Implementation>& implementations = registered_implementations();

    int passed = 0;
    int total = 0;
    std::vector<std::string> failures;

    for (const auto& implementation : implementations) {
        std::cout << "\n=== Testing " << implementation.name << " ===" << std::endl;

        for (const auto& test : tests) {
            ++total;
            const TestResult result = test.func(implementation.func);

            if (result.passed) {
                ++passed;
                std::cout << "  [PASS] " << test.name << std::endl;
                continue;
            }

            std::cout << "  [FAIL] " << test.name << std::endl;
            if (!result.message.empty()) {
                std::cout << "         " << result.message << std::endl;
            }
            failures.push_back(implementation.name + "::" + test.name);
        }
    }

    std::cout << "\nResults: " << passed << "/" << total << " passed" << std::endl;

    if (failures.empty()) {
        return 0;
    }

    std::cout << "Failed tests:" << std::endl;
    for (const auto& failure : failures) {
        std::cout << "  - " << failure << std::endl;
    }
    return -1;
}

}  // namespace FFTTest
