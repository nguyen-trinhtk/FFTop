#include "fftop/system.h"

#include <gtest/gtest.h>
#include <string>

TEST(System, DetectsHost) {
    const auto sys = FFTop::detect_system_config();
    EXPECT_FALSE(sys.cpu.empty());
    EXPECT_GE(sys.cpu_threads, 1u);

#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_EQ(sys.simd, FFTop::SIMD::NEON);
#elif defined(__x86_64__) || defined(_M_X64)
    EXPECT_TRUE(sys.simd == FFTop::SIMD::Scalar || sys.simd == FFTop::SIMD::AVX2 ||
                sys.simd == FFTop::SIMD::AVX512);
#endif

#if defined(__APPLE__)
    // sysctl always reports these on macOS
    EXPECT_GT(sys.cpu_cores, 0u);
    EXPECT_GT(sys.l1d_bytes, 0u);
    EXPECT_GT(sys.l2_bytes, 0u);
#endif
    if (sys.l1d_bytes && sys.l2_bytes) EXPECT_LE(sys.l1d_bytes, sys.l2_bytes);
    if (!sys.cuda) EXPECT_EQ(sys.gpu.smem_per_block, 0u);

    const std::string text = FFTop::describe_system(sys);
    EXPECT_NE(text.find("cpu:"), std::string::npos);
    EXPECT_NE(text.find("cache:"), std::string::npos);
    EXPECT_NE(text.find("simd:"), std::string::npos);
    EXPECT_NE(text.find("openmp:"), std::string::npos);
    EXPECT_NE(text.find("nvidia:"), std::string::npos);
    EXPECT_NE(text.find("cuda:"), std::string::npos);
}
