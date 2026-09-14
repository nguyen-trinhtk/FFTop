#include "fftop/backend/backend_registry.h"
#include "fftop/backend/cpu/simd/arch.h"
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
        FFTPlan plan;
        plan.radix           = RadixPolicy::Radix2;
        plan.hardware_target = HardwareTarget::CPU;
        CPUPlanOptions cpu;
        cpu.traversal = traversal;
        expect_matches_dft(*make_backend(plan, system_config(), cpu), kPowersOfTwo);
    }
}

TEST(Radix4, MatchesDft) {
    for (auto traversal : kTraversals) {
        SCOPED_TRACE(name(traversal));
        FFTPlan plan;
        plan.radix           = RadixPolicy::Radix4;
        plan.hardware_target = HardwareTarget::CPU;
        CPUPlanOptions cpu;
        cpu.traversal = traversal;
        expect_matches_dft(*make_backend(plan, system_config(), cpu), kPowersOfFour);
    }
}

// Catches a digit-reversal or output-ordering mismatch that a DFT comparison
// alone can hide behind a plausible-looking spectrum.
TEST(Radix4, AgreesWithRadix2) {
    FFTPlan plan2;
    plan2.radix           = RadixPolicy::Radix2;
    plan2.hardware_target = HardwareTarget::CPU;
    auto radix2 = make_backend(plan2);

    FFTPlan plan4;
    plan4.radix           = RadixPolicy::Radix4;
    plan4.hardware_target = HardwareTarget::CPU;
    auto radix4 = make_backend(plan4);

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
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;
        CPUPlanOptions serial_cpu;
        serial_cpu.execution = Execution::Serial;
        auto serial = make_backend(plan, system_config(), serial_cpu);

        CPUPlanOptions parallel_cpu;
        parallel_cpu.execution = Execution::Parallel;
        auto parallel = make_backend(plan, system_config(), parallel_cpu);

        expect_close(run(*parallel, input, Direction::Forward),
                     run(*serial, input, Direction::Forward));
    }
}

TEST(CPUBackend, DefaultExecutionFollowsOpenMp) {
    const auto input = ramp(256);
    FFTPlan plan;
    plan.size            = 256;
    plan.radix           = RadixPolicy::Radix4;
    plan.hardware_target = HardwareTarget::CPU;

    SystemConfig sys = system_config();
    sys.openmp       = false;
    auto from_sys = make_backend(plan, sys);
    CPUPlanOptions serial;
    serial.execution = Execution::Serial;
    auto explicit_serial = make_backend(plan, sys, serial);
    expect_close(run(*from_sys, input, Direction::Forward),
                 run(*explicit_serial, input, Direction::Forward));

    sys.openmp = true;
    from_sys   = make_backend(plan, sys);
    CPUPlanOptions parallel;
    parallel.execution = Execution::Parallel;
    auto explicit_parallel = make_backend(plan, sys, parallel);
    expect_close(run(*from_sys, input, Direction::Forward),
                 run(*explicit_parallel, input, Direction::Forward));
}

TEST(CPUBackend, AdaptiveAgreesWithScalar) {
    const auto input = mixed(64);
    SystemConfig scalar_sys = system_config();
    scalar_sys.simd = SIMD::Scalar;
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;
        auto adaptive = make_backend(plan);

        CPUPlanOptions scalar_cpu;
        scalar_cpu.execution = Execution::Serial;
        auto scalar = make_backend(plan, scalar_sys, scalar_cpu);

        expect_close(run(*adaptive, input, Direction::Forward),
                     run(*scalar, input, Direction::Forward),
                     radix == RadixPolicy::Radix4 ? "radix4" : "radix2");
    }
}

TEST(MakeBackend, AutoFallsBackToAvailableBackend) {
    auto backend = make_backend(FFTPlan{});
    ASSERT_TRUE(backend);
    EXPECT_TRUE(backend->is_available());
}

TEST(MakeBackend, ExplicitGpuRequiresDevice) {
    FFTPlan plan;
    plan.hardware_target = HardwareTarget::GPU;
    try {
        auto backend = make_backend(plan);
        EXPECT_TRUE(backend->is_available());
        EXPECT_NE(backend->name(), "CPU");
    } catch (const std::runtime_error&) {
        SUCCEED();
    }
}

#if defined(FFTOP_HAS_AVX512_KERNEL)
TEST(CPUBackend, AVX512AgreesWithScalar) {
    if (system_config().simd != SIMD::AVX512) GTEST_SKIP() << "host is not AVX-512";
    const auto input = mixed(64);
    SystemConfig avx512_sys = system_config();
    avx512_sys.simd = SIMD::AVX512;
    SystemConfig scalar_sys = system_config();
    scalar_sys.simd = SIMD::Scalar;
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;

        CPUPlanOptions serial;
        serial.execution = Execution::Serial;
        auto avx512 = make_backend(plan, avx512_sys, serial);
        auto scalar = make_backend(plan, scalar_sys, serial);

        expect_close(run(*avx512, input, Direction::Forward),
                     run(*scalar, input, Direction::Forward),
                     radix == RadixPolicy::Radix4 ? "radix4" : "radix2");
    }
}
#endif
