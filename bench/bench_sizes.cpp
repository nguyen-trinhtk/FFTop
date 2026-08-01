#include "utils.h"

namespace {

std::vector<FFTBench::BenchmarkRow> benchmark_input_sizes(const FFTCore::FFTFunc& fft) {
    // Mid-range sweep — locality wins show up here more than at 16M.
    const std::vector<std::size_t> sizes = {
        1u << 10,  // 1024
        1u << 12,  // 4096
        1u << 14,  // 16384
        1u << 16,  // 65536
        1u << 18,  // 262144
        1u << 20,  // 1048576
    };
    std::vector<FFTBench::BenchmarkRow> rows;
    rows.reserve(sizes.size());

    for (const std::size_t size : sizes) {
        const auto input = FFTBench::generate_random_complex(size);
        const int runs = 3;

        std::vector<FFTCore::Complex> warmup_output;
        fft(input, warmup_output);

        rows.push_back({
            "Input sizes",
            size,
            runs,
            FFTBench::benchmark_ms(fft, input, runs),
        });
    }

    return rows;
}

const FFTBench::BenchmarkRegistrar kInputSizes(
    "Input sizes",
    benchmark_input_sizes);

}  // namespace
