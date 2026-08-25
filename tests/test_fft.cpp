#include "fftop/backend/ibackend.h"
#include "fftop/backend/backend_registry.h"
#include "fftop/plan/plan.h"
#include "fftop/plan/planner.h"
#include "reference/dft.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {

using FFTop::Direction;
using FFTop::Execution;
using FFTop::RadixPolicy;
using FFTop::Real;
using FFTop::Traversal;

// A single radix only tiles sizes that are a power of it, so radix-4 covers a
// subset of the radix-2 sizes until mixed radix exists.
constexpr std::size_t kPowersOfTwo[]  = {1, 2, 4, 8, 16, 32, 64};
constexpr std::size_t kPowersOfFour[] = {1, 4, 16, 64, 256};

constexpr Traversal kTraversals[] = {Traversal::Iterative, Traversal::Recursive};

constexpr Real kTolerance = 1e-12;

FFTop::Buffer impulse(std::size_t n) {
    FFTop::Buffer x(n, {0, 0});
    if (n > 0) x[0] = {1, 0};
    return x;
}

FFTop::Buffer ramp(std::size_t n) {
    FFTop::Buffer x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = {Real(i), 0};
    return x;
}

// Impulse and ramp are purely real, so their spectra are conjugate-symmetric and
// can hide a real/imaginary mix-up in a SIMD kernel. This one cannot.
FFTop::Buffer mixed(std::size_t n) {
    FFTop::Buffer x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = {std::cos(Real(i) * Real(0.7)), std::sin(Real(i) * Real(1.3)) + Real(0.25)};
    return x;
}

// Scaled by the largest expected magnitude so one tolerance holds as N grows.
Real relative_error(const FFTop::Buffer& got, const FFTop::Buffer& want) {
    if (got.size() != want.size()) return std::numeric_limits<Real>::infinity();
    Real error = 0;
    Real scale = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        error = std::max(error, std::abs(got[i] - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    return scale > 0 ? error / scale : error;
}

FFTop::Buffer execute(FFTop::IBackend& backend, const FFTop::Buffer& input, Direction dir) {
    FFTop::FFTPlan plan;
    plan.size      = input.size();
    plan.direction = dir;
    FFTop::Buffer output;
    backend.execute(plan, input, output);
    return output;
}

// The inverse pass flips the twiddle sign and the radix-4 quarter turn, so both
// directions have to be checked to pin those signs down.
void expect_matches_dft(FFTop::IBackend& backend, const FFTop::Buffer& input) {
    for (Direction dir : {Direction::Forward, Direction::Inverse}) {
        auto want = FFTop::Ref::dft(input, dir);
        // FFTop leaves the inverse unnormalised; the reference divides by N.
        if (dir == Direction::Inverse)
            for (auto& z : want) z *= Real(input.size());

        EXPECT_LT(relative_error(execute(backend, input, dir), want), kTolerance)
            << backend.name() << " N=" << input.size()
            << (dir == Direction::Forward ? " forward" : " inverse");
    }
}

std::unique_ptr<FFTop::CPUBackend> make_cpu(RadixPolicy radix, Traversal traversal,
                                            Execution execution = Execution::Serial) {
    FFTop::FFTPlan plan;
    plan.radix     = radix;
    plan.traversal = traversal;
    plan.execution = execution;
    return FFTop::make_cpu_backend(plan);
}

}  // namespace

TEST(IBackend, FftMatchesDft) {
    for (auto& backend : FFTop::all_backends()) {
        if (!backend->is_available()) continue;
        SCOPED_TRACE(backend->name());
        for (std::size_t n : kPowersOfTwo) {
            expect_matches_dft(*backend, impulse(n));
            expect_matches_dft(*backend, ramp(n));
            expect_matches_dft(*backend, mixed(n));
        }
    }
}

TEST(Radix2, MatchesDft) {
    for (Traversal traversal : kTraversals) {
        auto backend = make_cpu(RadixPolicy::Radix2, traversal);
        SCOPED_TRACE(traversal == Traversal::Iterative ? "iterative" : "recursive");
        for (std::size_t n : kPowersOfTwo) {
            expect_matches_dft(*backend, impulse(n));
            expect_matches_dft(*backend, ramp(n));
            expect_matches_dft(*backend, mixed(n));
        }
    }
}

TEST(Radix4, MatchesDft) {
    for (Traversal traversal : kTraversals) {
        auto backend = make_cpu(RadixPolicy::Radix4, traversal);
        SCOPED_TRACE(traversal == Traversal::Iterative ? "iterative" : "recursive");
        for (std::size_t n : kPowersOfFour) {
            expect_matches_dft(*backend, impulse(n));
            expect_matches_dft(*backend, ramp(n));
            expect_matches_dft(*backend, mixed(n));
        }
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
            EXPECT_LT(relative_error(execute(*radix4, input, dir), execute(*radix2, input, dir)),
                      kTolerance)
                << "N=" << n;
        }
    }
}

TEST(CPUBackend, ParallelMatchesSerial) {
    const auto input = ramp(256);  // 4^4, so both radices tile it
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        auto serial   = make_cpu(radix, Traversal::Iterative, Execution::Serial);
        auto parallel = make_cpu(radix, Traversal::Iterative, Execution::Parallel);
        EXPECT_LT(relative_error(execute(*parallel, input, Direction::Forward),
                                 execute(*serial, input, Direction::Forward)),
                  kTolerance);
    }
}

TEST(Planner, PrefersRadix4WhenSizeIsPowerOfFour) {
    FFTop::Planner planner;
    EXPECT_EQ(planner.make_plan(256, {}).radix, RadixPolicy::Radix4);
    EXPECT_EQ(planner.make_plan(32, {}).radix, RadixPolicy::Radix2);
}
