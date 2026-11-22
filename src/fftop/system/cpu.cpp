#include "fftop/system/detect.h"
#include "fftop/backend/cpu/simd/kernels.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <utility>

// Include libraries to query cpuid depending on CPU arch/model
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32)
#include <sys/utsname.h>  // uname()
#endif

#if defined(__linux__)
#include <unistd.h>  // sysconf() cache sizes
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFTOP_X86 1
#if defined(_MSC_VER)  // MSVC
#include <intrin.h>
#else
#include <cpuid.h>  // GCC or Clang
#endif
#endif

namespace FFTop {
namespace {
std::string trim(std::string s) {  // trim whitespace
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

#if defined(FFTOP_X86)
void cpuid(int leaf, int subleaf, unsigned regs[4]) {
    // Leaf in EAX, subleaf in ECX
    // Return regs = [EAX, EBX, ECX, EDX]
#if defined(_MSC_VER)  // Windows
    __cpuidex(reinterpret_cast<int*>(regs), leaf, subleaf);
#else  // others
    __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
#endif
}

std::string x86_brand() {
    unsigned r[4]{};
    cpuid(static_cast<int>(0x80000000u), 0, r);  // highest "leave" it can answer
    if (r[0] < 0x80000004u) return {};  // too-old CPUs doesn't support querying for model name
    char buf[49]{};
    for (int i = 0; i < 3; ++i) {
        // name are sliced into 3 leaves
        cpuid(static_cast<int>(0x80000002u) + i, 0, r);
        std::memcpy(buf + i * 16, r, 16);  // 16 char per leave
    }
    return trim(buf);
}

// Read XCR0
// to later check if OS saves
// SIMD registers on context switch
std::uint64_t xcr0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else  // manually call ASM instruction
    // input: ecx=0 (read)
    // output: [edx|eax]
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<std::uint64_t>(edx) << 32) | eax;
#endif
}

SIMD x86_simd() {
    // Again highest leave
    unsigned r[4]{};
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 1) return SIMD::Scalar;
    // Check OS register saving and AVX support
    cpuid(1, 0, r);
    const bool fma     = r[2] & (1u << 12);
    const bool osxsave = r[2] & (1u << 27);
    const bool avx     = r[2] & (1u << 28);
    const auto xcr     = osxsave ? xcr0() : 0ull;
    // Check AVX2/AVX-512
    bool avx2    = false;
    bool avx512f = false;
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        avx2    = r[1] & (1u << 5);
        avx512f = r[1] & (1u << 16);
    }
    // Check if OS saves widest-register on context switch
    // Also require FMA
    if (avx512f && avx && fma && (xcr & 0xE6) == 0xE6) return SIMD::AVX512;
    if (avx2 && avx && fma && (xcr & 0x6) == 0x6) return SIMD::AVX2;
    return SIMD::Scalar;  // Fallback
}

#endif

#if defined(__APPLE__)
// Query macOS's sys control
std::string sysctl_str(const char* name) {
    size_t n = 0;
    // Prelim: check answer size in bytes
    if (sysctlbyname(name, nullptr, &n, nullptr, 0) != 0 || n == 0) return {};  // failed or empty
    // Fetch answer
    std::string s(n, '\0');  // prealloc n bytes
    if (sysctlbyname(name, s.data(), &n, nullptr, 0) != 0) return {};
    if (!s.empty() && s.back() == '\0') s.pop_back();  // drop null terminator
    return trim(s);
}

std::uint64_t sysctl_u64(const char* sysctl_key) {
    std::uint64_t v = 0;  // 32bit answer will fill low bytes
    size_t n        = sizeof v;
    if (sysctlbyname(sysctl_key, &v, &n, nullptr, 0) != 0) return 0;  // missing/error
    return v;
}

#endif

#if defined(__linux__)
// For Linux, read /proc/cpuinfo/
std::string proc_cpuinfo_model() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) return {};  // error opening file

    // Read each line
    std::string line, fallback;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');  // only care ab line with :
        if (colon == std::string::npos) continue;
        const auto key = trim(line.substr(0, colon));
        const auto val = trim(line.substr(colon + 1));

        // Read CPU model name
        if (key == "model name" && !val.empty()) return val;
        if ((key == "Hardware" || key == "cpu model") && fallback.empty())
            // fallback as some doesn't expose model name
            fallback = val;
    }
    return fallback;
}

// Num physical cores
std::size_t proc_cpuinfo_cores() {
    std::ifstream in("/proc/cpuinfo");
    if (!in) return 0;
    std::set<std::pair<long, long>> cores;
    long phys = 0;
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = trim(line.substr(0, colon));
        const auto val = trim(line.substr(colon + 1));
        try {
            if (key == "physical id")
                phys = std::stol(val);
            else if (key == "core id")
                cores.insert({phys, std::stol(val)});
        } catch (...) {
        }
    }
    return cores.size();
}

std::string read_first_line(const std::string& path) {
    std::ifstream in(path);
    std::string s;
    std::getline(in, s);
    return trim(s);
}

// String to bytes ( e.g. "32K", "1024K", "16M", "512" -> bytes)
std::size_t parse_size(const std::string& s) {
    std::size_t v = 0;
    std::size_t i = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
        v = v * 10 + static_cast<std::size_t>(s[i++] - '0');
    if (i < s.size()) {
        switch (std::toupper(static_cast<unsigned char>(s[i]))) {
            case 'K':
                v <<= 10;
                break;
            case 'M':
                v <<= 20;
                break;
            case 'G':
                v <<= 30;
                break;
            default:
                break;
        }
    }
    return v;
}

// Cache sizes seen by cpu0
CacheSizes sysfs_caches() {
    CacheSizes c;
    for (int i = 0; i < 16; ++i) {
        const std::string dir =
            "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        const std::string level = read_first_line(dir + "level");
        if (level.empty()) break;
        const std::string type = read_first_line(dir + "type");
        const std::size_t size = parse_size(read_first_line(dir + "size"));
        if (level == "1" && type == "Data")
            c.l1d = size;
        else if (level == "2" && type != "Instruction")
            c.l2 = size;
        else if (level == "3" && type != "Instruction")
            c.l3 = size;
    }
    return c;
}
#endif
}  // namespace

std::string detect_cpu_name() {
#if defined(FFTOP_X86)
    auto x86_name = x86_brand();
    if (!x86_name.empty()) return x86_name;
#endif
#if defined(__linux__)
    auto linux_from_proc = proc_cpuinfo_model();
    if (!linux_from_proc.empty()) return linux_from_proc;
#endif
#if defined(__APPLE__)
    auto apple_brand = sysctl_str("machdep.cpu.brand_string");
    if (!apple_brand.empty()) return apple_brand;
    auto apple_model = sysctl_str("hw.model");
    if (!apple_model.empty()) return "Apple Silicon (" + apple_model + ")";
#endif
#if !defined(_WIN32)
    utsname u{};
    if (uname(&u) == 0) return u.machine;
#endif
    return "unknown";
}

std::size_t detect_core_count() {
#if defined(__APPLE__)
    return static_cast<std::size_t>(sysctl_u64("hw.physicalcpu"));
#elif defined(__linux__)
    return proc_cpuinfo_cores();
#else
    return 0;
#endif
}

CacheSizes detect_cache_sizes() {
    CacheSizes c;
#if defined(__APPLE__)
    // Apple Silicon: prefer performance core (perflevel0)
    c.l1d = static_cast<std::size_t>(sysctl_u64("hw.perflevel0.l1dcachesize"));
    c.l2  = static_cast<std::size_t>(sysctl_u64("hw.perflevel0.l2cachesize"));
    // fallback to any cores
    if (c.l1d == 0) c.l1d = static_cast<std::size_t>(sysctl_u64("hw.l1dcachesize"));
    if (c.l2 == 0) c.l2 = static_cast<std::size_t>(sysctl_u64("hw.l2cachesize"));
    c.l3 = static_cast<std::size_t>(sysctl_u64("hw.l3cachesize"));

#elif defined(__linux__)
    c = sysfs_caches();
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    // glibc fallback
    if (c.l1d == 0) c.l1d = static_cast<std::size_t>(std::max(0L, sysconf(_SC_LEVEL1_DCACHE_SIZE)));
    if (c.l2 == 0) c.l2 = static_cast<std::size_t>(std::max(0L, sysconf(_SC_LEVEL2_CACHE_SIZE)));
    if (c.l3 == 0) c.l3 = static_cast<std::size_t>(std::max(0L, sysconf(_SC_LEVEL3_CACHE_SIZE)));
#endif
#endif
    return c;
}

SIMD detect_simd() {
// Use widest supported SIMD kernel
#if defined(FFTOP_X86)
    const SIMD hw = x86_simd();
#if defined(FFTOP_AVX512)
    if (hw == SIMD::AVX512) return SIMD::AVX512;
#endif
#if defined(FFTOP_AVX2)
    if (hw == SIMD::AVX2 || hw == SIMD::AVX512) return SIMD::AVX2;
#endif
    return SIMD::Scalar;
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(FFTOP_NEON)
    return SIMD::NEON;
#else
    return SIMD::Scalar;
#endif
#else
    return SIMD::Scalar;
#endif
}
}  // namespace FFTop
