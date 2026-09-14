#pragma once

#include "fftop/backend/backend_registry.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"
#include "support/check.h"

#include <memory>
#include <vector>

namespace FFTop::Test {

constexpr Traversal kTraversals[] = {Traversal::Iterative, Traversal::Recursive};

inline const char* name(Traversal traversal) {
    return traversal == Traversal::Iterative ? "iterative" : "recursive";
}

inline std::vector<std::unique_ptr<IBackend>> all_backends() {
    std::vector<std::unique_ptr<IBackend>> out;

    FFTPlan cpu;
    cpu.hardware_target = HardwareTarget::CPU;
    out.push_back(make_backend(cpu));

#if defined(FFTOP_ENABLE_CUDA)
    FFTPlan gpu;
    gpu.hardware_target = HardwareTarget::GPU;
    try {
        auto backend = make_backend(gpu);
        if (backend->is_available()) out.push_back(std::move(backend));
    } catch (const std::runtime_error&) {
    }
#endif
    return out;
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
