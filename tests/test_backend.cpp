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
    SystemConfig scalar_sys = system_config();
    scalar_sys.simd = SIMD::Scalar;
    const std::size_t sizes[] = {4, 16, 64, 256};
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;
        auto adaptive = make_backend(plan);

        CPUPlanOptions scalar_cpu;
        scalar_cpu.execution = Execution::Serial;
        auto scalar = make_backend(plan, scalar_sys, scalar_cpu);
        const char* which = radix == RadixPolicy::Radix4 ? "radix4" : "radix2";

        for (std::size_t n : sizes) {
            const auto input = mixed(n);
            for (Direction dir : {Direction::Forward, Direction::Inverse}) {
                expect_close(run(*adaptive, input, dir), run(*scalar, input, dir),
                             std::string(which) + " N=" + std::to_string(n));
            }
        }
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

TEST(GPUBackend, KernelStrategiesMatchDft) {
    const GPUKernelStrategy strategies[] = {
        GPUKernelStrategy::CooleyTukeyGlobal,
        GPUKernelStrategy::StockhamGlobal,
        GPUKernelStrategy::StockhamShared,
    };
    const std::size_t sizes[] = {1, 2, 4, 8, 16, 32, 64, 256, 512};

    for (auto strategy : strategies) {
        FFTPlan plan;
        plan.hardware_target = HardwareTarget::GPU;
        plan.radix           = RadixPolicy::Radix2;
        plan.gpu_strategy    = strategy;
        try {
            auto backend = make_backend(plan);
            if (!backend || !backend->is_available()) continue;
            SCOPED_TRACE(backend->name());
            expect_matches_dft(*backend, sizes);
        } catch (const std::runtime_error&) {
            SUCCEED();
            return;
        }
    }
}

TEST(GPUBackend, KernelStrategiesAgree) {
    const auto input = mixed(256);
    FFTPlan base;
    base.hardware_target = HardwareTarget::GPU;
    base.radix = RadixPolicy::Radix2;

    try {
        FFTPlan ct = base;
        ct.gpu_strategy = GPUKernelStrategy::CooleyTukeyGlobal;
        auto ct_backend = make_backend(ct);

        FFTPlan sg = base;
        sg.gpu_strategy = GPUKernelStrategy::StockhamGlobal;
        auto sg_backend = make_backend(sg);

        FFTPlan ss = base;
        ss.gpu_strategy = GPUKernelStrategy::StockhamShared;
        auto ss_backend = make_backend(ss);

        const auto ref = run(*ct_backend, input, Direction::Forward);
        expect_close(run(*sg_backend, input, Direction::Forward), ref, "stockham-global");
        expect_close(run(*ss_backend, input, Direction::Forward), ref, "stockham-shared");

        const auto big = mixed(1024);
        const auto big_ref = run(*ct_backend, big, Direction::Forward);
        expect_close(run(*sg_backend, big, Direction::Forward), big_ref, "stockham-global N=1024");
        expect_close(run(*ss_backend, big, Direction::Forward), big_ref, "stockham-shared N=1024");
    } catch (const std::runtime_error&) {
        SUCCEED();
    }
}

#if defined(FFTOP_HAS_AVX2_KERNEL)
TEST(CPUBackend, AVX2AgreesWithScalar) {
    const auto host = system_config().simd;
    if (host != SIMD::AVX2 && host != SIMD::AVX512) GTEST_SKIP() << "host is not AVX2";
    SystemConfig avx2_sys = system_config();
    avx2_sys.simd = SIMD::AVX2;
    SystemConfig scalar_sys = system_config();
    scalar_sys.simd = SIMD::Scalar;
    const std::size_t sizes[] = {4, 16, 64, 256};
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;

        CPUPlanOptions serial;
        serial.execution = Execution::Serial;
        auto avx2   = make_backend(plan, avx2_sys, serial);
        auto scalar = make_backend(plan, scalar_sys, serial);
        const char* which = radix == RadixPolicy::Radix4 ? "radix4" : "radix2";

        for (std::size_t n : sizes) {
            const auto input = mixed(n);
            for (Direction dir : {Direction::Forward, Direction::Inverse}) {
                expect_close(run(*avx2, input, dir), run(*scalar, input, dir),
                             std::string(which) + " N=" + std::to_string(n));
            }
        }
    }
}
#endif

#if defined(FFTOP_HAS_AVX512_KERNEL)
TEST(CPUBackend, AVX512AgreesWithScalar) {
    if (system_config().simd != SIMD::AVX512) GTEST_SKIP() << "host is not AVX-512";
    SystemConfig avx512_sys = system_config();
    avx512_sys.simd = SIMD::AVX512;
    SystemConfig scalar_sys = system_config();
    scalar_sys.simd = SIMD::Scalar;
    const std::size_t sizes[] = {4, 16, 64, 256};
    for (RadixPolicy radix : {RadixPolicy::Radix2, RadixPolicy::Radix4}) {
        FFTPlan plan;
        plan.radix           = radix;
        plan.hardware_target = HardwareTarget::CPU;

        CPUPlanOptions serial;
        serial.execution = Execution::Serial;
        auto avx512 = make_backend(plan, avx512_sys, serial);
        auto scalar = make_backend(plan, scalar_sys, serial);
        const char* which = radix == RadixPolicy::Radix4 ? "radix4" : "radix2";

        for (std::size_t n : sizes) {
            const auto input = mixed(n);
            for (Direction dir : {Direction::Forward, Direction::Inverse}) {
                expect_close(run(*avx512, input, dir), run(*scalar, input, dir),
                             std::string(which) + " N=" + std::to_string(n));
            }
        }
    }
}
#endif
