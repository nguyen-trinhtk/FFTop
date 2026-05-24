#include "utils.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>

namespace FFTBench {
namespace {

std::vector<BenchmarkCase>& benchmark_registry() {
    static std::vector<BenchmarkCase> benchmarks;
    return benchmarks;
}

std::vector<Implementation>& implementation_registry() {
    static std::vector<Implementation> implementations;
    return implementations;
}

std::string timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    const std::tm local_time = *std::localtime(&time);

    std::ostringstream out;
    out << std::put_time(&local_time, "%y-%m-%d %H:%M:%S");
    return out.str();
}

std::string markdown_table(
    const std::string& benchmark_name,
    const std::vector<BenchmarkRow>& rows) {
    std::ostringstream out;
    out << "| Benchmark | N | Runs | Avg ms |\n";
    out << "|---|---:|---:|---:|\n";

    for (const auto& row : rows) {
        out << "| " << benchmark_name
            << " | " << row.size
            << " | " << row.runs
            << " | " << std::fixed << std::setprecision(3) << row.avg_ms
            << " |\n";
    }

    return out.str();
}

}  // namespace

BenchmarkRegistrar::BenchmarkRegistrar(std::string name, BenchmarkFunction func) {
    benchmark_registry().push_back({std::move(name), std::move(func)});
}

ImplementationRegistrar::ImplementationRegistrar(std::string name, FFTCore::FFTFunc func) {
    implementation_registry().push_back({std::move(name), std::move(func)});
}

const std::vector<BenchmarkCase>& registered_benchmarks() {
    return benchmark_registry();
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

double benchmark_ms(
    const FFTCore::FFTFunc& fft,
    const std::vector<FFTCore::Complex>& input,
    int runs) {
    using Clock = std::chrono::steady_clock;

    const auto start = Clock::now();
    for (int run = 0; run < runs; ++run) {
        (void)fft(input);
    }
    const auto end = Clock::now();

    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / runs;
}

int run_all_benchmarks() {
    const auto& implementations = registered_implementations();
    const auto& benchmarks = registered_benchmarks();

    std::ostringstream report;
    report << "# Benchmarks\n";
    report << "Timestamp: " << timestamp_now() << "\n";

    for (const auto& implementation : implementations) {
        report << "### " << implementation.name << "\n";

        for (const auto& benchmark : benchmarks) {
            const auto rows = benchmark.func(implementation.func);
            report << markdown_table(benchmark.name, rows);
        }

        report << "\n";
    }

    std::cout << report.str();

    std::ofstream file("bench/results/bench.md", std::ios::out | std::ios::trunc);
    if (!file) {
        std::cerr << "Failed to open bench/results/bench.md for writing." << std::endl;
        return 1;
    }
    file << report.str();

    return 0;
}

}  // namespace FFTBench
