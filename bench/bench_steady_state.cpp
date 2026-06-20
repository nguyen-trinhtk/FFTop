#include "utils.h"

namespace {

std::vector<FFTBench::BenchmarkRow> benchmark_steady_state(const FFTCore::FFTFunc& fft) {
    const std::vector<std::size_t> sizes = {16777216};
    std::vector<FFTBench::BenchmarkRow> rows;
    rows.reserve(sizes.size());

    for (const std::size_t size : sizes) {
        const auto input = FFTBench::generate_random_complex(size);
        const int runs = 3;

        std::vector<FFTCore::Complex> warmup_output;
        fft(input, warmup_output);

        rows.push_back({
            "Steady state",
            size,
            runs,
            FFTBench::benchmark_ms(fft, input, runs),
        });
    }

    return rows;
}

const FFTBench::BenchmarkRegistrar kSteadyState(
    "Steady state",
    benchmark_steady_state);

}  // namespace
