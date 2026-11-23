#include "fftop/backend/cpu/cpu.h"
#include "fftop/math/twiddle.h"
#include "fftop/plan/planner.h"
#include "support/check.h"

#include <gtest/gtest.h>

#include <complex>
#include <stdexcept>

using namespace FFTop;

namespace {
constexpr double PI = 3.14159265358979323846;

// A made-up machine, so the heuristics are tested independently of the host
SystemConfig fake_system() {
    SystemConfig sys;
    sys.cpu_threads = 16;
    sys.cpu_cores = 8;
    sys.l1d_bytes = 32 * 1024;
    sys.l2_bytes = 1 << 20;  // four-step tile = 512 KiB / 16 B = 32768
    sys.simd = SIMD::Scalar;
    sys.openmp = true;
    return sys;
}

// Device limits as cudaGetDeviceProperties reports them
GPUInfo t4() {
    GPUInfo gpu;
    gpu.smem_per_block = 48 * 1024;
    gpu.smem_per_block_optin = 64 * 1024;
    gpu.smem_per_sm = 64 * 1024;
    gpu.max_threads_per_block = 1024;
    gpu.max_threads_per_sm = 1024;
    return gpu;
}

GPUInfo a100() {
    GPUInfo gpu = t4();
    gpu.smem_per_block_optin = 163 * 1024;
    gpu.smem_per_sm = 164 * 1024;
    gpu.reserved_smem_per_block = 1024;
    gpu.max_threads_per_sm = 2048;
    return gpu;
}

GPUInfo rtx4090() {
    GPUInfo gpu = a100();
    gpu.smem_per_block_optin = 99 * 1024;
    gpu.smem_per_sm = 100 * 1024;
    gpu.max_threads_per_sm = 1536;
    return gpu;
}

SystemConfig fake_gpu_system() {
    SystemConfig sys = fake_system();
    sys.cuda = true;
    sys.nvidia_gpu = true;
    sys.gpu = t4();
    return sys;
}

const CPUConfig& cpu_of(const PlanSpec& spec) { return std::get<CPUConfig>(spec.config); }
}  // namespace

// ---- CPUPlanner ------------------------------------------------------------

TEST(CPUPlanner, TinySizesUseDft) {
    const CPUPlanner planner(fake_system(), {});
    EXPECT_EQ(planner.plan(2).algorithm, CPUAlgorithm::DFT);
    EXPECT_EQ(planner.plan(8).algorithm, CPUAlgorithm::DFT);  // dft_max = 8
    EXPECT_EQ(planner.plan(16).algorithm, CPUAlgorithm::Iterative);

    EstimateParams params;
    params.dft_max = 4;
    EXPECT_EQ(CPUPlanner(fake_system(), params).plan(8).algorithm, CPUAlgorithm::Iterative);
}

TEST(CPUPlanner, PrefersRadix4WhenSizeIsPowerOfFour) {
    const CPUPlanner planner(fake_system(), {});
    EXPECT_EQ(planner.plan(256).radix, Radix::Radix4);
    EXPECT_EQ(planner.plan(512).radix, Radix::Radix2);
}

TEST(CPUPlanner, FourStepAboveL2Budget) {
    const CPUPlanner planner(fake_system(), {});
    EXPECT_EQ(planner.tile_threshold(), 32768u);
    EXPECT_EQ(planner.plan(32768).split, 0u);
    const CPUConfig big = planner.plan(65536);
    EXPECT_EQ(big.algorithm, CPUAlgorithm::Iterative);
    EXPECT_EQ(big.split, 256u);  // balanced 256 x 256
    EXPECT_NO_THROW(validate(big, 65536));
}

TEST(CPUPlanner, FourStepSplitIsBalancedAndCapped) {
    const CPUPlanner planner(fake_system(), {});
    EXPECT_EQ(planner.split_for(std::size_t(1) << 20), 1024u);   // 1024 x 1024
    EXPECT_EQ(planner.split_for(std::size_t(1) << 21), 2048u);   // N1 = 1024 <= N2
    EXPECT_EQ(planner.split_for(std::size_t(1) << 40), 32768u);  // capped by tile_threshold()

    SystemConfig small = fake_system();
    small.l2_bytes = 64 * 1024;  // tile = 2048
    EXPECT_EQ(CPUPlanner(small, {}).split_for(std::size_t(1) << 30), 2048u);
}

TEST(CPUPlanner, TileThresholdFollowsL2) {
    SystemConfig sys = fake_system();
    sys.l2_bytes = 4 << 20;
    EXPECT_EQ(CPUPlanner(sys, {}).tile_threshold(), 131072u);
    sys.l2_bytes = 0;  // unknown -> default_l2 (1 MiB)
    EXPECT_EQ(CPUPlanner(sys, {}).tile_threshold(), 32768u);
}

TEST(CPUPlanner, ThreadsOnlyWhenWorthIt) {
    SystemConfig sys = fake_system();
    EXPECT_EQ(CPUPlanner(sys, {}).plan(std::size_t(1) << 20).threads, 8);  // physical cores
    EXPECT_EQ(CPUPlanner(sys, {}).plan(64).threads, 1);
    sys.cpu_cores = 0;  // unknown -> logical threads
    EXPECT_EQ(CPUPlanner(sys, {}).plan(std::size_t(1) << 20).threads, 16);
    sys.openmp = false;
    EXPECT_EQ(CPUPlanner(sys, {}).plan(std::size_t(1) << 20).threads, 1);
}

TEST(CPUPlanner, UsesHostSimd) {
    SystemConfig sys = fake_system();
    sys.simd = SIMD::NEON;
    EXPECT_EQ(CPUPlanner(sys, {}).plan(1024).simd, SIMD::NEON);
}

// ---- GPUPlanner ------------------------------------------------------------

// Longest column L shows as the largest N one pass covers, and as the block (4 L threads)
TEST(GPUPlanner, ColumnFromDeviceLimits) {
    const auto passes = [](const GPUInfo& gpu, unsigned log2_n) {
        return GPUPlanner(gpu).plan(std::size_t(1) << log2_n).passes;
    };
    // T4: 256 fits the 64 KiB opt-in but leaves 1 block/SM -> 128
    EXPECT_EQ(passes(t4(), 7), 1u);
    EXPECT_EQ(passes(t4(), 8), 2u);
    EXPECT_EQ(GPUPlanner(t4()).plan(1024).block_dim, 512u);
    // A100: 512 needs 2048 threads/block -> 256 (2 blocks/SM)
    EXPECT_EQ(passes(a100(), 8), 1u);
    EXPECT_EQ(passes(a100(), 9), 2u);
    EXPECT_EQ(GPUPlanner(a100()).plan(1024).block_dim, 1024u);
    // RTX 4090: 256 leaves 1 block/SM by smem -> 128
    EXPECT_EQ(passes(rtx4090(), 8), 2u);
    GPUInfo small = t4();  // threads cap L at 32
    small.max_threads_per_block = 128;
    EXPECT_EQ(passes(small, 5), 1u);
    EXPECT_EQ(passes(small, 6), 2u);
    EXPECT_EQ(passes(GPUInfo{}, 7), 1u);  // unknown -> 128
    EXPECT_EQ(passes(GPUInfo{}, 8), 2u);
}

TEST(GPUPlanner, FewestPasses) {
    const GPUPlanner planner(t4());  // L = 128
    const auto passes = [&](unsigned log2_n) {
        return planner.plan(std::size_t(1) << log2_n).passes;
    };
    EXPECT_EQ(passes(1), 1u);
    EXPECT_EQ(passes(14), 2u);  // 128 x 128
    EXPECT_EQ(passes(15), 3u);  // 32 x 32 x 32
    EXPECT_EQ(passes(21), 3u);
    EXPECT_EQ(passes(22), 4u);
}

TEST(GPUPlanner, PlansValidate) {
    for (const GPUInfo& gpu : {t4(), a100(), rtx4090(), GPUInfo{}}) {
        for (std::size_t n = 1; n <= (std::size_t(1) << 24); n *= 2) {
            const GPUConfig cfg = GPUPlanner(gpu).plan(n);
            EXPECT_EQ(cfg.kernel, GPUKernel::StockhamSmem);
            EXPECT_EQ(cfg.radix, Radix::Radix2);
            EXPECT_TRUE(cfg.pinned_staging);
            EXPECT_NO_THROW(validate(cfg, n, gpu)) << "N=" << n;
        }
    }
}

// ---- Planner -----------------------------------------------------------------

TEST(Planner, AutoPicksGpuOnlyForLargeSizes) {
    const Planner gpu(fake_gpu_system());
    EXPECT_EQ(gpu.make_spec(std::size_t(1) << 20, HardwareTarget::Auto).target(),
              HardwareTarget::GPU);
    EXPECT_EQ(gpu.make_spec(1024, HardwareTarget::Auto).target(), HardwareTarget::CPU);

    const Planner cpu_only(fake_system());
    EXPECT_EQ(cpu_only.make_spec(std::size_t(1) << 20, HardwareTarget::Auto).target(),
              HardwareTarget::CPU);
}

TEST(Planner, ExplicitTargets) {
    const Planner gpu(fake_gpu_system());
    EXPECT_EQ(gpu.make_spec(std::size_t(1) << 20, HardwareTarget::CPU).target(),
              HardwareTarget::CPU);
    EXPECT_EQ(gpu.make_spec(16, HardwareTarget::GPU).target(), HardwareTarget::GPU);

    const Planner cpu_only(fake_system());
    EXPECT_THROW(cpu_only.make_spec(16, HardwareTarget::GPU), std::runtime_error);
}

TEST(Planner, DescribeShowsTheChoice) {
    const Planner planner(fake_system());
    EXPECT_EQ(describe(planner.make_spec(8, HardwareTarget::CPU)), "CPU dft N=8");
    EXPECT_EQ(describe(planner.make_spec(1024, HardwareTarget::CPU)),
              "CPU iterative r4 scalar t1 N=1024");
    EXPECT_EQ(cpu_of(planner.make_spec(65536, HardwareTarget::CPU)).split, 256u);
}

TEST(Planner, CachesPlans) {
    PlanCache cache;
    const Planner planner(system_config(), {}, &cache);

    const auto a = planner.plan(64, HardwareTarget::CPU);
    const auto b = planner.plan(64, HardwareTarget::CPU);
    EXPECT_EQ(a, b);  // built once
    EXPECT_EQ(cache.entries(), 1u);

    planner.plan(64, HardwareTarget::Auto);  // separate key, same size
    EXPECT_EQ(cache.entries(), 2u);

    // One plan runs both directions
    const Buffer x = FFTop::Test::mixed(64);
    Buffer fwd, inv;
    a->execute(fwd, x, Direction::Forward);
    a->execute(inv, x, Direction::Inverse);
    FFTop::Test::expect_close(fwd, FFTop::Test::expected_dft(x, Direction::Forward), "forward");
    FFTop::Test::expect_close(inv, FFTop::Test::expected_dft(x, Direction::Inverse), "inverse");

    cache.clear();
    EXPECT_EQ(cache.entries(), 0u);
    EXPECT_NE(planner.plan(64, HardwareTarget::CPU), a);  // rebuilt
}

TEST(Planner, WithoutCacheBuildsEachTime) {
    const Planner planner(system_config());
    EXPECT_NE(planner.plan(64, HardwareTarget::CPU), planner.plan(64, HardwareTarget::CPU));
}

// ---- Twiddles ----------------------------------------------------------------

TEST(Twiddles, CpuCacheBuildsOnce) {
    const auto t = CPU::twiddles(64, 4);
    ASSERT_TRUE(t);
    EXPECT_EQ(t->n, 64u);
    EXPECT_EQ(t->radix, 4u);
    EXPECT_EQ(t->w.size(), 63u);
    EXPECT_EQ(t, CPU::twiddles(64, 4));  // same shared table, not rebuilt
    EXPECT_NE(t, CPU::twiddles(64, 2));
}

TEST(Twiddles, StageTableLayout) {
    // w[s - 1 + (m - 1) s + k] == W_{R s}^{m k} = cis(-2pi m k / (R s))
    for (std::size_t R : {2u, 4u}) {
        const std::size_t n = 1024;
        const Twiddles st = make_twiddles(n, R);
        ASSERT_EQ(st.w.size(), n - 1);
        for (std::size_t s = 1; s < n; s *= R)
            for (std::size_t m = 1; m < R; ++m)
                for (std::size_t k = 0; k < s; ++k) {
                    const Complex want = std::polar(1.0, -2.0 * PI * double(m * k) / double(R * s));
                    const Complex got = st.w[s - 1 + (m - 1) * s + k];
                    ASSERT_NEAR(got.real(), want.real(), 1e-15)
                        << "R=" << R << " s=" << s << " m=" << m << " k=" << k;
                    ASSERT_NEAR(got.imag(), want.imag(), 1e-15)
                        << "R=" << R << " s=" << s << " m=" << m << " k=" << k;
                }
    }
}

TEST(Twiddles, RejectsBadRadix) {
    EXPECT_THROW(make_twiddles(16, 0), std::invalid_argument);
    EXPECT_THROW(make_twiddles(16, 1), std::invalid_argument);
}
