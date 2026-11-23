#include "fftop/math/fft_math.h"
#include "fftop/plan/plan.h"
#include "fftop/plan/planner.h"
#include "fftop/backend/cpu/simd/kernels.h"
#include "support/cpu.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace FFTop;
using namespace FFTop::Test;

// ---- CPU algorithms --------------------------------------------------------

TEST(CPU, IterativeAndRecursiveMatchDft) {
    for (auto algorithm : {CPUAlgorithm::Iterative, CPUAlgorithm::Recursive}) {
        SCOPED_TRACE(to_string(algorithm));
        expect_config_matches_dft(cpu_config(algorithm, Radix::Radix2), POWERS_OF_TWO);
        expect_config_matches_dft(cpu_config(algorithm, Radix::Radix4), POWERS_OF_FOUR);
    }
}

TEST(CPU, DftMatchesDft) {
    expect_config_matches_dft(cpu_config(CPUAlgorithm::DFT), POWERS_OF_TWO);
}

TEST(CPU, FourStepMatchesDft) {
    const std::size_t sizes[] = {32, 64, 128, 256, 1024};
    for (std::size_t split : {2u, 4u, 16u}) {
        SCOPED_TRACE("split=" + std::to_string(split));
        expect_config_matches_dft(four_step(split), sizes);
        expect_config_matches_dft(four_step(split, Radix::Radix4), sizes);
        CPUConfig recursive = four_step(split);  // split is independent of the traversal
        recursive.algorithm = CPUAlgorithm::Recursive;
        expect_config_matches_dft(recursive, sizes);
    }
}

TEST(CPU, LargeFourStepAgreesWithIterative) {
    const std::size_t n = std::size_t(1) << 14;
    const auto input = mixed(n);
    auto split = build(n, four_step(128, Radix::Radix4));  // 128 x 128 rows, radix 2 there
    auto iterative = build(n, cpu_config(CPUAlgorithm::Iterative, Radix::Radix4));
    for (Direction dir : {Direction::Forward, Direction::Inverse})
        expect_close(run(*split, input, dir), run(*iterative, input, dir), split->name());
}

// Catches a digit-reversal or output-ordering mismatch that a DFT comparison
// alone can hide behind a plausible-looking spectrum.
TEST(CPU, Radix4AgreesWithRadix2) {
    for (std::size_t n : POWERS_OF_FOUR) {
        auto radix2 = build(n, cpu_config(CPUAlgorithm::Iterative, Radix::Radix2));
        auto radix4 = build(n, cpu_config(CPUAlgorithm::Iterative, Radix::Radix4));
        const auto input = mixed(n);
        for (Direction dir : {Direction::Forward, Direction::Inverse}) {
            expect_close(run(*radix4, input, dir), run(*radix2, input, dir),
                         "N=" + std::to_string(n));
        }
    }
}

TEST(CPU, ParallelMatchesSerial) {
    const std::size_t n = 4096;  // 4^6, so both radices tile it
    const auto input = mixed(n);
    const CPUConfig configs[] = {
        cpu_config(CPUAlgorithm::Iterative, Radix::Radix2),
        cpu_config(CPUAlgorithm::Iterative, Radix::Radix4),
        cpu_config(CPUAlgorithm::Recursive, Radix::Radix2),
        four_step(64, Radix::Radix4),
    };
    for (CPUConfig serial : configs) {
        CPUConfig parallel = serial;
        parallel.threads = 4;
        auto s = build(n, serial);
        auto p = build(n, parallel);
        expect_close(run(*p, input, Direction::Forward), run(*s, input, Direction::Forward),
                     p->name());
    }
}

TEST(CPU, HostSimdAgreesWithScalar) {
    const auto input = mixed(64);
    for (Radix radix : {Radix::Radix2, Radix::Radix4}) {
        CPUConfig scalar = cpu_config(CPUAlgorithm::Iterative, radix);
        scalar.simd = SIMD::Scalar;
        auto host = build(64, cpu_config(CPUAlgorithm::Iterative, radix));
        auto ref = build(64, scalar);
        expect_close(run(*host, input, Direction::Forward), run(*ref, input, Direction::Forward),
                     host->name());
    }
}

#if defined(FFTOP_AVX2)
TEST(CPU, AVX2AgreesWithScalar) {
    const auto host = system_config().simd;
    if (host != SIMD::AVX2 && host != SIMD::AVX512) GTEST_SKIP() << "host is not AVX2";
    const auto input = mixed(64);
    for (Radix radix : {Radix::Radix2, Radix::Radix4}) {
        CPUConfig avx2 = cpu_config(CPUAlgorithm::Iterative, radix);
        avx2.simd = SIMD::AVX2;
        CPUConfig scalar = avx2;
        scalar.simd = SIMD::Scalar;
        expect_close(run(*build(64, avx2), input, Direction::Forward),
                     run(*build(64, scalar), input, Direction::Forward),
                     radix == Radix::Radix4 ? "radix4" : "radix2");
    }
}
#endif

#if defined(FFTOP_AVX512)
TEST(CPU, AVX512AgreesWithScalar) {
    if (system_config().simd != SIMD::AVX512) GTEST_SKIP() << "host is not AVX-512";
    const auto input = mixed(64);
    for (Radix radix : {Radix::Radix2, Radix::Radix4}) {
        CPUConfig avx512 = cpu_config(CPUAlgorithm::Iterative, radix);
        avx512.simd = SIMD::AVX512;
        CPUConfig scalar = avx512;
        scalar.simd = SIMD::Scalar;
        expect_close(run(*build(64, avx512), input, Direction::Forward),
                     run(*build(64, scalar), input, Direction::Forward),
                     radix == Radix::Radix4 ? "radix4" : "radix2");
    }
}
#endif

TEST(CPU, NameDescribesConfig) {
    CPUConfig cfg = four_step(16, Radix::Radix4, 2);
    cfg.simd = SIMD::Scalar;
    EXPECT_EQ(build(256, cfg)->name(), "CPU/iterative split=16 r4 scalar t2");
    EXPECT_EQ(build(8, cpu_config(CPUAlgorithm::DFT))->name(), "CPU/dft");
}

TEST(CPU, RejectsWrongInputSize) {
    auto backend = build(16, cpu_config(CPUAlgorithm::Iterative));
    Buffer out;
    EXPECT_THROW(backend->execute(out, Buffer(8), Direction::Forward), std::invalid_argument);
}

TEST(CPU, HasNoDevicePath) {
    auto backend = build(16, cpu_config(CPUAlgorithm::Iterative));
    Complex dummy[16];
    EXPECT_THROW(backend->execute_device(dummy, dummy, Direction::Forward), std::logic_error);
}

// ---- Forced configs that can't work must fail --------------------------------

TEST(Validate, InvalidCpuConfigsThrow) {
    using E = std::invalid_argument;
    EXPECT_THROW(build(32, cpu_config(CPUAlgorithm::Iterative, Radix::Radix4)), E);  // 32 != 4^k
    EXPECT_THROW(build(8, cpu_config(CPUAlgorithm::Recursive, Radix::Radix4)), E);
    EXPECT_THROW(build(64, four_step(64)), E);  // split must be < N
    EXPECT_THROW(build(64, four_step(128)), E);
    EXPECT_THROW(build(64, four_step(6)), E);  // not a power of 2
    EXPECT_THROW(build(64, four_step(1)), E);
    EXPECT_THROW(build(48, four_step(16)), E);  // N not a power of 2
    CPUConfig split_dft = cpu_config(CPUAlgorithm::DFT);
    split_dft.split = 4;
    EXPECT_THROW(build(16, split_dft), E);

    CPUConfig no_threads = cpu_config(CPUAlgorithm::Iterative);
    no_threads.threads = 0;
    EXPECT_THROW(build(16, no_threads), E);
}

TEST(Validate, InvalidGpuConfigsThrow) {
    using E = std::invalid_argument;
    GPUInfo t4;
    t4.max_threads_per_block = 1024;
    t4.smem_per_block = 48 * 1024;
    t4.smem_per_block_optin = 64 * 1024;
    const GPUInfo unknown;

    GPUConfig ok;
    ok.passes = 2;
    EXPECT_NO_THROW(validate(ok, 1024, t4));
    EXPECT_NO_THROW(validate(ok, 1, unknown));

    GPUConfig stockham_r4 = ok;
    stockham_r4.radix = Radix::Radix4;
    EXPECT_THROW(validate(stockham_r4, 64, t4), E);
    stockham_r4.kernel = GPUKernel::StockhamGmem;
    EXPECT_THROW(validate(stockham_r4, 64, t4), E);

    GPUConfig ct_r4 = ok;
    ct_r4.kernel = GPUKernel::CooleyTukeyGmem;
    ct_r4.radix = Radix::Radix4;
    EXPECT_NO_THROW(validate(ct_r4, 64, t4));
    EXPECT_THROW(validate(ct_r4, 32, t4), E);

    for (unsigned block : {0u, 16u, 48u, 2048u}) {
        GPUConfig bad = ok;
        bad.block_dim = block;
        EXPECT_THROW(validate(bad, 1024, t4), E) << "block=" << block;
    }
    for (unsigned passes : {0u, 11u}) {  // log2 1024 = 10
        GPUConfig bad = ok;
        bad.passes = passes;
        EXPECT_THROW(validate(bad, 1024, t4), E) << "passes=" << passes;
    }

    // Column of c points: 2 x c x 8 x 16 B, against the opt-in limit
    GPUInfo small_smem = t4;
    small_smem.smem_per_block_optin = 16 * 1024;
    GPUConfig big = ok;
    EXPECT_NO_THROW(validate(big, std::size_t(1) << 14, t4));  // 128 x 128: 32 KiB
    EXPECT_THROW(validate(big, std::size_t(1) << 14, small_smem), E);
    EXPECT_NO_THROW(validate(big, std::size_t(1) << 16, t4));  // 256 x 256: 64 KiB
    EXPECT_THROW(validate(big, std::size_t(1) << 17, t4), E);  // 512 x 256
    big.passes = 3;  // 64 x 64 x 32
    EXPECT_NO_THROW(validate(big, std::size_t(1) << 17, t4));
    big.passes = 1;  // one 1024-point column
    EXPECT_THROW(validate(big, 1024, t4), E);

    GPUConfig big_block = ok;
    big_block.block_dim = 1024;  // smem doesn't scale with block
    EXPECT_NO_THROW(validate(big_block, 1024, t4));

    EXPECT_THROW(validate(ok, 48, t4), E);  // N not a power of 2
}

TEST(MakePlan, GpuPlanWithoutCudaThrows) {
#if defined(FFTOP_ENABLE_CUDA)
    if (system_config().cuda) GTEST_SKIP() << "CUDA device present";
#endif
    EXPECT_THROW(build(16, GPUConfig{}), std::runtime_error);
}

// ---- GPU (CUDA builds with a device) -------------------------------------------

#if defined(FFTOP_ENABLE_CUDA)
namespace {
// The planner's config for n, with the kernel forced
GPUConfig gpu_config(GPUKernel kernel, std::size_t n) {
    GPUConfig cfg = GPUPlanner(system_config().gpu).plan(n);
    cfg.kernel = kernel;
    return cfg;
}
constexpr GPUKernel KERNELS[] = {
    GPUKernel::CooleyTukeyGmem,
    GPUKernel::StockhamGmem,
    GPUKernel::StockhamSmem,
};
}  // namespace

TEST(GPUPlan, KernelsMatchDft) {
    if (!system_config().cuda) GTEST_SKIP() << "no CUDA device";
    for (std::size_t n : {1, 2, 4, 8, 16, 32, 64, 256, 512, 1024}) {
        for (auto kernel : KERNELS) expect_config_matches_dft(gpu_config(kernel, n), {n});
        // Every pass count whose columns fit 32 KiB (uneven spreads included)
        GPUConfig cfg = gpu_config(GPUKernel::StockhamSmem, n);
        const unsigned log2_n = Math::log2_floor(n);
        for (unsigned passes = 1; passes <= log2_n; ++passes) {
            if (Math::ceil_div(log2_n, passes) > 7) continue;
            cfg.passes = passes;
            expect_config_matches_dft(cfg, {n});
        }
    }
}

TEST(GPUPlan, KernelsAgree) {
    if (!system_config().cuda) GTEST_SKIP() << "no CUDA device";
    for (std::size_t n : {std::size_t(1024), std::size_t(1) << 16, std::size_t(1) << 20}) {
        const auto input = mixed(n);
        const auto ref =
            run(*build(n, gpu_config(GPUKernel::CooleyTukeyGmem, n)), input, Direction::Forward);
        for (auto kernel : {GPUKernel::StockhamGmem, GPUKernel::StockhamSmem}) {
            auto backend = build(n, gpu_config(kernel, n));
            expect_close(run(*backend, input, Direction::Forward), ref,
                         backend->name() + " N=" + std::to_string(n));
        }
    }
}

TEST(GPUPlan, PinnedAndPageableAgree) {
    if (!system_config().cuda) GTEST_SKIP() << "no CUDA device";
    const std::size_t n = 4096;
    const auto input = mixed(n);
    GPUConfig pinned = gpu_config(GPUKernel::StockhamSmem, n);
    GPUConfig pageable = pinned;
    pageable.pinned_staging = false;
    auto a = build(n, pinned);
    auto b = build(n, pageable);
    for (int repeat = 0; repeat < 2; ++repeat)  // buffers are reused across calls
        expect_close(run(*a, input, Direction::Forward), run(*b, input, Direction::Forward),
                     b->name());
}
#endif
