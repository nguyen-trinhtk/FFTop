#include "fftop/system.h"

#include <gtest/gtest.h>
#include <string>

TEST(System, DetectsHost) {
    const auto sys = FFTop::detect_system_config();
    EXPECT_FALSE(sys.cpu.empty());
    EXPECT_GE(sys.cpu_threads, 1u);

#if defined(__aarch64__) || defined(_M_ARM64)
    EXPECT_EQ(sys.simd, FFTop::Simd::Neon);
#elif defined(__x86_64__) || defined(_M_X64)
    EXPECT_TRUE(sys.simd == FFTop::Simd::Sse2 || sys.simd == FFTop::Simd::Avx2 ||
                sys.simd == FFTop::Simd::Avx512);
#endif

    if (sys.kernel_simd == FFTop::Simd::Avx2) {
        EXPECT_TRUE(sys.simd == FFTop::Simd::Avx2 || sys.simd == FFTop::Simd::Avx512);
    } else if (sys.kernel_simd == FFTop::Simd::Neon) {
        EXPECT_EQ(sys.simd, FFTop::Simd::Neon);
    } else {
        EXPECT_EQ(sys.kernel_simd, FFTop::Simd::Scalar);
    }

    const std::string text = FFTop::describe_system(sys);
    EXPECT_NE(text.find("cpu:"), std::string::npos);
    EXPECT_NE(text.find("simd:"), std::string::npos);
    EXPECT_NE(text.find("openmp:"), std::string::npos);
    EXPECT_NE(text.find("nvidia:"), std::string::npos);
}
