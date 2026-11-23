#pragma once

// Helpers for tests that build plans from hand-made configs
// (the planner is bypassed on purpose).

#include "fftop/plan/plan.h"
#include "fftop/system.h"
#include "support/check.h"

#include <memory>

namespace FFTop::Test {

// CPUConfig on this host's SIMD
inline CPUConfig cpu_config(CPUAlgorithm algorithm, Radix radix = Radix::Radix2, int threads = 1) {
    CPUConfig cfg;
    cfg.algorithm = algorithm;
    cfg.radix = radix;
    cfg.simd = system_config().simd;
    cfg.threads = threads;
    return cfg;
}

// Iterative with a four-step split
inline CPUConfig four_step(std::size_t split, Radix radix = Radix::Radix2, int threads = 1) {
    CPUConfig cfg = cpu_config(CPUAlgorithm::Iterative, radix, threads);
    cfg.split = split;
    return cfg;
}

template <class Config>
std::unique_ptr<Plan> build(std::size_t n, const Config& cfg) {
    return build_plan(PlanSpec{n, cfg});
}

inline Buffer run(const Plan& plan, const Buffer& input, Direction dir) {
    Buffer output;
    plan.execute(output, input, dir);
    return output;
}

// One plan per size, checked against the DFT on every signal, both directions
template <class Config, std::size_t N>
void expect_config_matches_dft(const Config& cfg, const std::size_t (&sizes)[N]) {
    for (std::size_t n : sizes) {
        auto plan = build(n, cfg);
        for (auto signal : SIGNALS) {
            expect_matches_dft([&](const Buffer& x, Direction d) { return run(*plan, x, d); },
                               signal(n), plan->name());
        }
    }
}

}  // namespace FFTop::Test
