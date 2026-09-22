#include "fftop/backend/backend_registry.h"
#include "fftop/math/fft_math.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"
#include "fftop/types.h"
#include "reference/dft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(FFTOP_ENABLE_OPENMP)
#include <omp.h>
#endif

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/gpu_backend.h"
#include "fftop/backend/gpu/gpu_utils.h"
#include "fftop/math/twiddle.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int    kMinRepeats = 20;
constexpr int    kMaxRepeats = 5000;
constexpr double kTargetMs   = 100.0;

struct BenchSpec {
    std::string backend;
    std::string traversal;
    std::string radix;
    std::string simd;
    std::string execution;
    int         threads = 1;
    std::size_t size = 0;
};

enum class GPUBenchVariant { CooleyTukey, StockhamGlobal, StockhamShared };

void usage() {
    std::cerr
        << "fftop_bench --system\n"
        << "fftop_bench [--dry-run] [--no-header]\n"
        << "  stdin: backend,traversal,radix,simd,execution,threads,size\n"
        << "  stdout: schema-3 CSV rows\n";
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n";  break;
        default:   out += c;      break;
        }
    }
    return out;
}

void write_system_json() {
    const auto& cfg = FFTop::system_config();
    std::cout << "{\n"
              << "  \"schema\": 3,\n"
              << "  \"cpu\": \"" << json_escape(cfg.cpu) << "\",\n"
              << "  \"cpu_threads\": " << cfg.cpu_threads << ",\n"
              << "  \"simd\": \"" << FFTop::to_string(cfg.simd) << "\",\n"
              << "  \"openmp\": " << (cfg.openmp ? "true" : "false") << ",\n"
              << "  \"nvidia_gpu\": " << (cfg.nvidia_gpu ? "true" : "false")
              << ",\n"
              << "  \"cuda\": " << (cfg.cuda ? "true" : "false") << "\n"
              << "}\n";
}

bool parse_spec(const std::string& line, BenchSpec& spec, std::string& err) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    if (cols.size() != 7) {
        err = "expected 7 columns, got " + std::to_string(cols.size());
        return false;
    }
    spec.backend   = cols[0];
    spec.traversal = cols[1];
    spec.radix     = cols[2];
    spec.simd      = cols[3];
    spec.execution = cols[4];
    try {
        spec.threads = std::stoi(cols[5]);
        spec.size    = static_cast<std::size_t>(std::stoull(cols[6]));
    } catch (...) {
        err = "bad threads or size";
        return false;
    }
    if (spec.threads < 1) {
        err = "threads must be >= 1";
        return false;
    }
    if (spec.size == 0) {
        err = "size must be > 0";
        return false;
    }
    return true;
}

FFTop::SIMD parse_simd(const std::string& s, std::string& err) {
    if (s == "scalar" || s == "-") return FFTop::SIMD::Scalar;
    if (s == "avx2") return FFTop::SIMD::AVX2;
    if (s == "avx512") return FFTop::SIMD::AVX512;
    if (s == "neon") return FFTop::SIMD::NEON;
    err = "unknown simd '" + s + "'";
    return FFTop::SIMD::Scalar;
}

bool parse_plan(const BenchSpec& spec, FFTop::FFTPlan& plan, FFTop::SystemConfig& sys,
                FFTop::CPUPlanOptions& cpu, GPUBenchVariant& gpu_variant,
                std::string& err) {
    plan.size = spec.size;
    plan.hardware_target = spec.backend == "GPU" ? FFTop::HardwareTarget::GPU
                                                : FFTop::HardwareTarget::CPU;
    if (spec.radix == "4")
        plan.radix = FFTop::RadixPolicy::Radix4;
    else
        plan.radix = FFTop::RadixPolicy::Radix2;
    cpu.traversal = spec.traversal == "recursive"
                        ? FFTop::Traversal::Recursive
                        : FFTop::Traversal::Iterative;
    cpu.execution = spec.execution == "parallel" ? FFTop::Execution::Parallel
                                                 : FFTop::Execution::Serial;
    sys           = FFTop::system_config();
    sys.simd      = parse_simd(spec.simd, err);
    if (spec.backend == "GPU") {
        if (spec.traversal == "cooley-tukey" || spec.traversal == "iterative") {
            gpu_variant = GPUBenchVariant::CooleyTukey;
        } else if (spec.traversal == "stockham-global") {
            gpu_variant = GPUBenchVariant::StockhamGlobal;
        } else if (spec.traversal == "stockham-shared") {
            gpu_variant = GPUBenchVariant::StockhamShared;
        } else {
            err = "unknown GPU traversal '" + spec.traversal + "'";
        }
    }
    return err.empty();
}

int choose_repeats(double one_ms) {
    if (!(one_ms > 0.0) || !std::isfinite(one_ms)) return kMaxRepeats;
    return std::clamp(static_cast<int>(std::lround(kTargetMs / one_ms)),
                      kMinRepeats, kMaxRepeats);
}

template <typename Fn>
double time_ms(const BenchSpec& spec, Fn&& fn) {
    fn();
    const auto t_one0 = Clock::now();
    fn();
    const double one_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_one0).count();
    const int repeats = choose_repeats(one_ms);

    const auto t0 = Clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() /
        repeats;

    std::cerr << spec.backend << ' ' << spec.traversal << " r" << spec.radix
              << ' ' << spec.simd << " t" << spec.threads << " N=" << spec.size
              << " repeats=" << repeats << ' ' << ms << " ms\n";
    return ms;
}

void print_row(const BenchSpec& spec, double ms, std::uint64_t global_mem_bytes) {
    const double bandwidth_gbps =
        (global_mem_bytes == 0 || !(ms > 0.0))
            ? 0.0
            : static_cast<double>(global_mem_bytes) / (ms * 1.0e6);
    const double throughput_gsamples =
        (spec.size == 0 || !(ms > 0.0)) ? 0.0 : static_cast<double>(spec.size) / (ms * 1.0e6);

    std::cout << spec.backend << ',' << spec.traversal << ',' << spec.radix
              << ',' << spec.simd << ',' << spec.execution << ','
              << spec.threads << ',' << spec.size << ',' << ms
              << ',' << global_mem_bytes << ',' << bandwidth_gbps
              << ',' << throughput_gsamples << '\n';
}

void apply_threads(int threads) {
#if defined(FFTOP_ENABLE_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
}

#if defined(FFTOP_ENABLE_CUDA)
double time_gpu_ms(const BenchSpec& spec, FFTop::GPUFFTBackend& gpu, const FFTop::FFTPlan& plan) {
    using FFTop::Complex;
    const std::size_t n = spec.size;
    FFTop::GPU::DeviceBuf d_src(n);
    FFTop::GPU::DeviceBuf d_a(n);
    FFTop::GPU::DeviceBuf d_b(gpu.needs_scratch() ? n : 0);
    FFTop::GPU::DeviceBuf d_W(n);

    FFTop::Buffer input(n, {1, 0});
    FFTop::GPU::check_cuda(
        cudaMemcpy(d_src.ptr, input.data(), n * sizeof(double2), cudaMemcpyHostToDevice),
        "bench H2D");
    const auto W = FFTop::default_twiddles().get(n);
    FFTop::GPU::check_cuda(
        cudaMemcpy(d_W.ptr, W->data(), n * sizeof(double2), cudaMemcpyHostToDevice),
        "bench twiddle H2D");

    auto restore = [&] {
        FFTop::GPU::check_cuda(
            cudaMemcpy(d_a.ptr, d_src.ptr, n * sizeof(double2), cudaMemcpyDeviceToDevice),
            "bench restore");
    };
    auto run = [&] {
        gpu.execute_device(plan, reinterpret_cast<Complex*>(d_a.ptr),
                           reinterpret_cast<Complex*>(d_b.ptr),
                           reinterpret_cast<const Complex*>(d_W.ptr));
    };

    restore();
    run();
    FFTop::GPU::check_cuda(cudaDeviceSynchronize(), "bench warmup");

    cudaEvent_t start, stop;
    FFTop::GPU::check_cuda(cudaEventCreate(&start), "cudaEventCreate");
    FFTop::GPU::check_cuda(cudaEventCreate(&stop), "cudaEventCreate");

    restore();
    FFTop::GPU::check_cuda(cudaEventRecord(start), "cudaEventRecord");
    run();
    FFTop::GPU::check_cuda(cudaEventRecord(stop), "cudaEventRecord");
    FFTop::GPU::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float one_ms = 0;
    FFTop::GPU::check_cuda(cudaEventElapsedTime(&one_ms, start, stop), "cudaEventElapsedTime");
    const int repeats = choose_repeats(one_ms);

    double total_ms = 0;
    for (int i = 0; i < repeats; ++i) {
        restore();
        FFTop::GPU::check_cuda(cudaEventRecord(start), "cudaEventRecord");
        run();
        FFTop::GPU::check_cuda(cudaEventRecord(stop), "cudaEventRecord");
        FFTop::GPU::check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0;
        FFTop::GPU::check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        total_ms += ms;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const double ms = total_ms / repeats;
    std::cerr << spec.backend << ' ' << spec.traversal << " r" << spec.radix
              << ' ' << spec.simd << " t" << spec.threads << " N=" << spec.size
              << " repeats=" << repeats << ' ' << ms << " ms\n";
    return ms;
}
#endif

bool run_spec(const BenchSpec& spec, bool dry, std::string& err) {
    if (spec.backend != "GPU" && spec.radix == "4" &&
        !FFTop::Math::is_power_of(spec.size, 4)) {
        err = "radix-4 requires N = 4^p";
        return false;
    }

    if (dry) {
        std::cerr << spec.backend << ',' << spec.traversal << ',' << spec.radix
                  << ',' << spec.simd << ',' << spec.execution << ','
                  << spec.threads << ',' << spec.size << '\n';
        return true;
    }

    apply_threads(spec.threads);

    if (spec.backend == "naive-dft") {
        FFTop::Buffer input(spec.size, {1, 0});
        const double ms = time_ms(spec, [&] { (void)FFTop::Ref::dft(input); });
        print_row(spec, ms, 0);
        return true;
    }

    if (spec.backend != "CPU" && spec.backend != "GPU") {
        err = "unknown backend '" + spec.backend + "'";
        return false;
    }

    FFTop::FFTPlan plan;
    FFTop::SystemConfig sys;
    FFTop::CPUPlanOptions cpu;
    GPUBenchVariant gpu_variant = GPUBenchVariant::CooleyTukey;
    if (!parse_plan(spec, plan, sys, cpu, gpu_variant, err)) return false;
    if (spec.backend == "GPU") {
        if (spec.radix != "2") {
            err = "GPU variants currently require radix=2";
            return false;
        }
        switch (gpu_variant) {
            case GPUBenchVariant::CooleyTukey:
                plan.gpu_strategy = FFTop::GPUKernelStrategy::CooleyTukeyGlobal;
                break;
            case GPUBenchVariant::StockhamGlobal:
                plan.gpu_strategy = FFTop::GPUKernelStrategy::StockhamGlobal;
                break;
            case GPUBenchVariant::StockhamShared:
                plan.gpu_strategy = FFTop::GPUKernelStrategy::StockhamShared;
                break;
        }
    }

    auto backend = FFTop::make_backend(plan, sys, cpu);
#if defined(FFTOP_ENABLE_CUDA)
    if (spec.backend == "GPU") {
        auto* gpu = dynamic_cast<FFTop::GPUFFTBackend*>(backend.get());
        if (!gpu) {
            err = "GPU backend unavailable";
            return false;
        }
        const double ms = time_gpu_ms(spec, *gpu, plan);
        print_row(spec, ms, gpu->estimated_global_bytes(spec.size));
        return true;
    }
#endif

    FFTop::Buffer input(spec.size, {1, 0});
    FFTop::Buffer output;
    const double ms = time_ms(spec, [&] { backend->execute(plan, input, output); });
    print_row(spec, ms, 0);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool dry       = false;
    bool system_only = false;
    bool no_header = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--system") {
            system_only = true;
        } else if (a == "--dry-run") {
            dry = true;
        } else if (a == "--no-header") {
            no_header = true;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::cerr << "unknown argument: " << a << '\n';
            usage();
            return 2;
        }
    }

    if (system_only) {
        write_system_json();
        return 0;
    }

    if (isatty(STDIN_FILENO)) {
        usage();
        return 2;
    }

    if (!dry) {
        std::cerr << FFTop::describe_system(FFTop::system_config());
        std::cerr << "repeats: target " << kTargetMs << " ms, clamp ["
                  << kMinRepeats << ", " << kMaxRepeats << "]\n";
    }
    if (!no_header && !dry) {
        std::cout << "# schema=3\n";
        std::cout << "backend,traversal,radix,simd,execution,threads,"
                     "size,ms,global_mem_bytes,effective_bandwidth_gbps,throughput_gsamples_s\n";
    }

    int rc = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        BenchSpec spec;
        std::string err;
        if (!parse_spec(line, spec, err)) {
            std::cerr << "error: " << line << " (" << err << ")\n";
            rc = 1;
            continue;
        }
        if (!run_spec(spec, dry, err)) {
            std::cerr << "skip: " << line << " (" << err << ")\n";
        }
    }
    return rc;
}
