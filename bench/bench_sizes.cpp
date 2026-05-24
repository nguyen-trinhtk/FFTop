#include "utils.h"

namespace {

std::vector<FFTBench::BenchmarkRow> benchmark_input_sizes(const FFTCore::FFTFunc& fft) {
    const std::vector<std::size_t> sizes = {64, /* 128, */ 256, /* 512, 1024,2048, */ 4096, /* 65536, */ /* 1048576 */};
    std::vector<FFTBench::BenchmarkRow> rows;
    rows.reserve(sizes.size());

    for (const std::size_t size : sizes) {
        const auto input = FFTBench::generate_random_complex(size);
        // const int runs = size <= 512 ? 5 : 3;
        const int runs = 10;

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
