#include "fftop/backend/ibackend.h"
#include "fftop/backend/backend_registry.h"
#include "fftop/plan/plan.h"
#include "reference/dft.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {

FFTop::Buffer impulse(std::size_t n) {
    FFTop::Buffer x(n, {0, 0});
    if (n > 0) x[0] = {1, 0};
    return x;
}

FFTop::Buffer ramp(std::size_t n) {
    FFTop::Buffer x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = {FFTop::Real(i), 0};
    return x;
}

FFTop::Real max_abs_error(const FFTop::Buffer& a, const FFTop::Buffer& b) {
    if (a.size() != b.size()) return std::numeric_limits<FFTop::Real>::infinity();
    FFTop::Real max_err = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    return max_err;
}

FFTop::Buffer execute(FFTop::IBackend& backend, const FFTop::Buffer& input) {
    FFTop::FFTPlan plan;
    plan.size = input.size();
    FFTop::Buffer output;
    backend.execute(plan, input, output);
    return output;
}

void expect_matches_dft(FFTop::IBackend& backend, const FFTop::Buffer& input) {
    const auto got  = execute(backend, input);
    const auto want = FFTop::Ref::dft(input);
    EXPECT_LT(max_abs_error(got, want), 1e-10) << backend.name() << " N=" << input.size();
}

std::unique_ptr<FFTop::CPUBackend> make_cpu(FFTop::Traversal traversal,
                                            FFTop::Execution execution) {
    FFTop::FFTPlan plan;
    plan.traversal = traversal;
    plan.execution = execution;
    return FFTop::make_cpu_backend(plan);
}

}  // namespace

TEST(IBackend, FftMatchesDft) {
    constexpr std::size_t sizes[] = {1, 2, 4, 8, 16, 32};

    for (auto& backend : FFTop::all_backends()) {
        if (!backend->is_available()) continue;
        SCOPED_TRACE(backend->name());
        for (std::size_t n : sizes) {
            expect_matches_dft(*backend, impulse(n));
            expect_matches_dft(*backend, ramp(n));
        }
    }
}

TEST(CPUBackend, RecursiveMatchesDft) {
    auto backend = make_cpu(FFTop::Traversal::Recursive, FFTop::Execution::Serial);
    for (std::size_t n : {1u, 2u, 4u, 8u, 16u, 32u}) {
        expect_matches_dft(*backend, impulse(n));
        expect_matches_dft(*backend, ramp(n));
    }
}

TEST(CPUBackend, ParallelMatchesSerial) {
    auto serial   = make_cpu(FFTop::Traversal::Iterative, FFTop::Execution::Serial);
    auto parallel = make_cpu(FFTop::Traversal::Iterative, FFTop::Execution::Parallel);
    const auto input = ramp(32);
    EXPECT_LT(max_abs_error(execute(*serial, input), execute(*parallel, input)), 1e-10);
}
