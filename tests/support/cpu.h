#pragma once

#include "fftop/backend/backend_registry.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"
#include "support/check.h"

#include <memory>

namespace FFTop::Test {

constexpr Traversal kTraversals[] = {Traversal::Iterative, Traversal::Recursive};

inline const char* name(Traversal traversal) {
    return traversal == Traversal::Iterative ? "iterative" : "recursive";
}

inline std::unique_ptr<CPUBackend> make_cpu(RadixPolicy radix, Traversal traversal,
                                            Execution execution = Execution::Serial,
                                            Simd simd = system_config().kernel_simd) {
    FFTPlan plan;
    plan.radix     = radix;
    plan.traversal = traversal;
    plan.execution = execution;
    return make_cpu_backend(plan, simd);
}

inline Buffer run(IBackend& backend, const Buffer& input, Direction dir) {
    FFTPlan plan;
    plan.size      = input.size();
    plan.direction = dir;
    Buffer output;
    backend.execute(plan, input, output);
    return output;
}

inline void expect_matches_dft(IBackend& backend, const Buffer& input) {
    expect_matches_dft([&](const Buffer& x, Direction d) { return run(backend, x, d); },
                       input, backend.name());
}

template <std::size_t N>
void expect_matches_dft(IBackend& backend, const std::size_t (&sizes)[N]) {
    for_each_input(sizes, [&](const Buffer& x) { expect_matches_dft(backend, x); });
}

}  // namespace FFTop::Test
