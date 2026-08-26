#include "fftop/system/detect.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
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
#if defined(_MSC_VER)
    __cpuidex(reinterpret_cast<int*>(regs), leaf, sub);
#else
    __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
}

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

Simd x86_simd() {
    unsigned r[4]{};
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return Simd::Scalar;

    cpuid(1, 0, r);
    const bool sse2    = r[3] & (1u << 26);
    const bool osxsave = r[2] & (1u << 27);
    const bool avx     = r[2] & (1u << 28);

    bool avx_os = false;
    bool avx512_os = false;
    if (osxsave) {
#if defined(_MSC_VER)
        const auto xcr0 = _xgetbv(0);
#else
        unsigned eax, edx;
        __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
        const auto xcr0 = (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
        avx_os    = (xcr0 & 0x6) == 0x6;
        avx512_os = (xcr0 & 0xE0) == 0xE0;
    }

    bool avx2 = false;
    bool avx512f = false;
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        avx2    = r[1] & (1u << 5);
        avx512f = r[1] & (1u << 16);
    }

    if (avx512f && avx && avx_os && avx512_os) return Simd::Avx512;
    if (avx2 && avx && avx_os) return Simd::Avx2;
    if (sse2) return Simd::Sse2;
    return Simd::Scalar;
}

#endif

#if defined(__APPLE__)

bool sysctl_u32(const char* name, std::uint32_t& out) {
    size_t n = sizeof(out);
    return sysctlbyname(name, &out, &n, nullptr, 0) == 0;
}

std::string sysctl_str(const char* name) {
    size_t n = 0;
    if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return {};
    std::string s(n, '\0');
    if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();
    return trim(s);
}

const char* apple_family_name(std::uint32_t family) {
    switch (family) {
    case 0x1b588bb3u: return "Apple M1";
    case 0xda33d83du: return "Apple M2";
    case 0x8765edeau: return "Apple A16";
    case 0xfa33415eu: return "Apple M3";
    case 0x5f4dea93u: return "Apple M3 Pro";
    case 0x72015832u: return "Apple M3 Max";
    case 0x6f5129acu: return "Apple M4";
    case 0x17d5b93au: return "Apple M4 Pro/Max";
    default:          return nullptr;
    }
}

#endif

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

}  // namespace

std::string detect_cpu_name() {
#if defined(FFTOP_X86)
    if (auto brand = x86_brand(); !brand.empty()) return brand;
#endif
    if (auto from_proc = proc_cpuinfo_model(); !from_proc.empty()) return from_proc;
#if defined(__APPLE__)
    if (auto brand = sysctl_str("machdep.cpu.brand_string"); !brand.empty()) return brand;
    std::uint32_t family = 0;
    if (sysctl_u32("hw.cpufamily", family)) {
        if (const char* name = apple_family_name(family)) {
            const auto model = sysctl_str("hw.model");
            return model.empty() ? std::string(name) : std::string(name) + " (" + model + ")";
        }
    }
    if (auto model = sysctl_str("hw.model"); !model.empty()) return "Apple Silicon (" + model + ")";
#endif
#if !defined(_WIN32)
    utsname u{};
    if (uname(&u) == 0) return u.machine;
#endif
    return "unknown";
}

Simd detect_simd() {
#if defined(FFTOP_X86)
    return x86_simd();
#elif defined(__aarch64__) || defined(_M_ARM64)
    return Simd::Neon;
#else
    return Simd::Scalar;
#endif
}

}  // namespace FFTop
