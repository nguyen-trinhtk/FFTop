#include "fftop/system/detect.h"
#include "fftop/backend/cpu/simd/arch.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32)
#include <sys/utsname.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFTOP_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace FFTop {
namespace {

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

#if defined(FFTOP_X86)

void cpuid(int leaf, int sub, unsigned regs[4]) {
    // x86 will expose all supported features in the CPUID leaves
#if defined(_MSC_VER) // Windows
    __cpuidex(reinterpret_cast<int*>(regs), leaf, sub);
#else // others
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
}

// x86 detection: brand
// Intel/AMD stash name in three CPUID leaves
std::string x86_brand() {
    unsigned r[4]{};
    cpuid(static_cast<int>(0x80000000u), 0, r);
    if (r[0] < 0x80000004u) return {};
    char buf[49]{};
    for (int i = 0; i < 3; ++i) {
        cpuid(static_cast<int>(0x80000002u) + i, 0, r);
        std::memcpy(buf + i * 16, r, 16);
    }
    return trim(buf);
}

// XCR0: which wide register files the OS will actually save on a context switch.
std::uint64_t xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

SIMD x86_simd() {
    unsigned r[4]{};
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return SIMD::Scalar;

    cpuid(1, 0, r);
    const bool osxsave = r[2] & (1u << 27);
    const bool avx     = r[2] & (1u << 28);
    const auto xcr     = osxsave ? xcr0() : 0ull;

    bool avx2 = false;
    bool avx512f = false;
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        avx2    = r[1] & (1u << 5);
        avx512f = r[1] & (1u << 16);
    }

    // Chip bit AND OS-enabled registers. Missing either → crash on first use.
    if (avx512f && avx && (xcr & 0xE6) == 0xE6) return SIMD::AVX512;
    if (avx2 && avx && (xcr & 0x6) == 0x6)      return SIMD::AVX2;
    return SIMD::Scalar;
}

#endif

#if defined(__APPLE__)

std::string sysctl_str(const char* name) {
    size_t n = 0;
    if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return {};
    std::string s(n, '\0');
    if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return trim(s);
}

#endif

#if defined(__linux__)

std::string proc_cpuinfo_model() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) return {};
    std::string line, fallback;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = trim(line.substr(0, colon));
        const auto val = trim(line.substr(colon + 1));
        if (key == "model name" && !val.empty()) return val;
        if ((key == "Hardware" || key == "cpu model") && fallback.empty()) fallback = val;
    }
    return fallback;
}

#endif

}  // namespace

std::string detect_cpu_name() {
#if defined(FFTOP_X86)
    if (auto brand = x86_brand(); !brand.empty()) return brand;
#endif
#if defined(__linux__)
    if (auto from_proc = proc_cpuinfo_model(); !from_proc.empty()) return from_proc;
#endif
#if defined(__APPLE__)
    if (auto brand = sysctl_str("machdep.cpu.brand_string"); !brand.empty()) return brand;
    if (auto model = sysctl_str("hw.model"); !model.empty()) return "Apple Silicon (" + model + ")";
#endif
#if !defined(_WIN32)
    utsname u{};
    if (uname(&u) == 0) return u.machine;
#endif
    return "unknown";
}

SIMD detect_simd() {
// Use widest supported SIMD kernel
#if defined(FFTOP_X86)
    const SIMD hw = x86_simd();
#if defined(FFTOP_HAS_AVX512_KERNEL)
    if (hw == SIMD::AVX512) return SIMD::AVX512;
#endif
#if defined(FFTOP_HAS_AVX2_KERNEL)
    if (hw == SIMD::AVX2 || hw == SIMD::AVX512) return SIMD::AVX2;
#endif
    return SIMD::Scalar;
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(FFTOP_HAS_NEON_KERNEL)
    return SIMD::NEON;
#else
    return SIMD::Scalar;
#endif
#else
    return SIMD::Scalar;
#endif
}
}  // namespace FFTop
