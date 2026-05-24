#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <fft/core/types.h>

namespace FFTBench {

struct BenchmarkRow {
    std::string label;
    std::size_t size;
    int runs;
    double avg_ms;
};

using BenchmarkFunction = std::function<std::vector<BenchmarkRow>(const FFTCore::FFTFunc&)>;

struct BenchmarkCase {
    std::string name;
    BenchmarkFunction func;
};

struct Implementation {
    std::string name;
    FFTCore::FFTFunc func;
};

class BenchmarkRegistrar {
public:
    BenchmarkRegistrar(std::string name, BenchmarkFunction func);
};

class ImplementationRegistrar {
public:
    ImplementationRegistrar(std::string name, FFTCore::FFTFunc func);
};

const std::vector<BenchmarkCase>& registered_benchmarks();
const std::vector<Implementation>& registered_implementations();

std::vector<FFTCore::Complex> generate_random_complex(std::size_t size);
double benchmark_ms(
    const FFTCore::FFTFunc& fft,
    const std::vector<FFTCore::Complex>& input,
    int runs);

int run_all_benchmarks();

}  // namespace FFTBench
