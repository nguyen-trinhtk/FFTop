#include "support/cpu.h"

#include <gtest/gtest.h>

using namespace FFTop;
using namespace FFTop::Test;

TEST(IBackend, FftMatchesDft) {
    for (auto& backend : all_backends()) {
        if (!backend->is_available()) continue;
        SCOPED_TRACE(backend->name());
        expect_matches_dft(*backend, kPowersOfTwo);
    }
}

TEST(Radix2, MatchesDft) {
    for (auto traversal : kTraversals) {
        SCOPED_TRACE(name(traversal));
        expect_matches_dft(*make_cpu(RadixPolicy::Radix2, traversal), kPowersOfTwo);
    }
}

TEST(Radix4, MatchesDft) {
    for (auto traversal : kTraversals) {
        SCOPED_TRACE(name(traversal));
        expect_matches_dft(*make_cpu(RadixPolicy::Radix4, traversal), kPowersOfFour);
    }
}

// Catches a digit-reversal or output-ordering mismatch that a DFT comparison
// alone can hide behind a plausible-looking spectrum.
TEST(Radix4, AgreesWithRadix2) {
    auto radix2 = make_cpu(RadixPolicy::Radix2, Traversal::Iterative);
    auto radix4 = make_cpu(RadixPolicy::Radix4, Traversal::Iterative);

    for (std::size_t n : kPowersOfFour) {
        const auto input = mixed(n);
        for (Direction dir : {Direction::Forward, Direction::Inverse}) {
            expect_close(run(*radix4, input, dir), run(*radix2, input, dir),
                         "N=" + std::to_string(n));
        }
    }
}

TEST(CPUBackend, ParallelMatchesSerial) {
    const auto input = ramp(256);  // 4^4, so both radices tile it
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        auto serial   = make_cpu(radix, Traversal::Iterative, Execution::Serial);
        auto parallel = make_cpu(radix, Traversal::Iterative, Execution::Parallel);
        expect_close(run(*parallel, input, Direction::Forward),
                     run(*serial, input, Direction::Forward));
    }
}

TEST(CPUBackend, AdaptiveAgreesWithScalar) {
    const auto input = mixed(64);
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        auto adaptive = make_cpu(radix, Traversal::Iterative);
        auto scalar   = make_cpu(radix, Traversal::Iterative, Execution::Serial, Simd::Scalar);
        expect_close(run(*adaptive, input, Direction::Forward),
                     run(*scalar, input, Direction::Forward),
                     radix == RadixPolicy::Radix4 ? "radix4" : "radix2");
    }
}
