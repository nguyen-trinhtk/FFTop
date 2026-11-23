#include "fftop/plan/plan.h"
#include "fftop/plan/planner.h"
#include "fftop/system.h"
#include "fftop/types.h"
#include "reference/dft.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(FFTOP_ENABLE_CUDA)
#include "fftop/backend/gpu/gpu.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int MIN_REPEATS = 20;
constexpr int MAX_REPEATS = 5000;
constexpr double TARGET_MS = 100.0;

struct BenchSpec {
    std::string backend;
    std::string traversal;
    std::string radix;
    std::string simd;
    std::string execution;
    int threads = 1;
    std::size_t size = 0;
};

void usage() {
    std::cerr << "fftop_bench --system\n"
              << "fftop_bench [--dry-run] [--no-header]\n"
              << "  stdin: backend,traversal,radix,simd,execution,threads,size\n"
              << "    backend:   CPU | GPU | auto (the planner's choice; other columns ignored) | "
                 "naive-dft\n"
              << "    traversal: CPU: iterative | recursive | dft | four-step (iterative + the "
                 "planner's split)\n"
              << "               GPU: cooley-tukey-gmem | stockham-gmem | stockham-smem\n"
              << "    threads:   used when execution=parallel (OpenMP num_threads, per run)\n"
              << "  stdout: schema-3 CSV rows\n";
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out += c;
                break;
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
              << "  \"cpu_cores\": " << cfg.cpu_cores << ",\n"
              << "  \"l1d_bytes\": " << cfg.l1d_bytes << ",\n"
              << "  \"l2_bytes\": " << cfg.l2_bytes << ",\n"
              << "  \"l3_bytes\": " << cfg.l3_bytes << ",\n"
              << "  \"simd\": \"" << FFTop::to_string(cfg.simd) << "\",\n"
              << "  \"openmp\": " << (cfg.openmp ? "true" : "false") << ",\n"
              << "  \"nvidia_gpu\": " << (cfg.nvidia_gpu ? "true" : "false") << ",\n"
              << "  \"cuda\": " << (cfg.cuda ? "true" : "false") << ",\n"
              << "  \"gpu_name\": \"" << json_escape(cfg.gpu.name) << "\",\n"
              << "  \"gpu_cc\": \"" << cfg.gpu.cc_major << '.' << cfg.gpu.cc_minor << "\",\n"
              << "  \"gpu_sm_count\": " << cfg.gpu.sm_count << ",\n"
              << "  \"gpu_smem_per_block\": " << cfg.gpu.smem_per_block << ",\n"
              << "  \"gpu_max_threads_per_block\": " << cfg.gpu.max_threads_per_block << "\n"
              << "}\n";
}

bool parse_spec(BenchSpec& spec, const std::string& line, std::string& err) {
    std::vector<std::string> cols;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) cols.push_back(cell);
    if (cols.size() != 7) {
        err = "expected 7 columns, got " + std::to_string(cols.size());
        return false;
    }
    spec.backend = cols[0];
    spec.traversal = cols[1];
    spec.radix = cols[2];
    spec.simd = cols[3];
    spec.execution = cols[4];
    try {
        spec.threads = std::stoi(cols[5]);
        spec.size = static_cast<std::size_t>(std::stoull(cols[6]));
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

// A forced config from a spec row (the planner is bypassed), or the planner's
// own choice for backend "auto". Launch shapes and the four-step tile come from
// the planners, so a row only names what it compares.
bool parse_spec(FFTop::PlanSpec& plan_spec, const BenchSpec& spec, std::string& err) {
    const FFTop::SystemConfig& sys = FFTop::system_config();
    const FFTop::EstimateParams params;
    if (spec.backend == "auto") {
        plan_spec = FFTop::Planner(sys, params).make_spec(spec.size, FFTop::HardwareTarget::Auto);
        return true;
    }

    plan_spec.size = spec.size;
    const FFTop::Radix radix = spec.radix == "4" ? FFTop::Radix::Radix4 : FFTop::Radix::Radix2;
    if (spec.backend == "GPU") {
        FFTop::GPUConfig cfg = FFTop::GPUPlanner(sys.gpu).plan(spec.size);
        cfg.radix = radix;
        if (spec.traversal == "cooley-tukey-gmem" || spec.traversal == "iterative") {
            cfg.kernel = FFTop::GPUKernel::CooleyTukeyGmem;
        } else if (spec.traversal == "stockham-gmem") {
            cfg.kernel = FFTop::GPUKernel::StockhamGmem;
        } else if (spec.traversal == "stockham-smem") {
            cfg.kernel = FFTop::GPUKernel::StockhamSmem;
        } else {
            err = "unknown GPU traversal '" + spec.traversal + "'";
        }
        plan_spec.config = cfg;
        return err.empty();
    }

    FFTop::CPUConfig cfg;
    cfg.radix = radix;
    cfg.simd = parse_simd(spec.simd, err);
    cfg.threads = spec.execution == "parallel" ? spec.threads : 1;
    if (spec.traversal == "iterative") {
        cfg.algorithm = FFTop::CPUAlgorithm::Iterative;
    } else if (spec.traversal == "recursive") {
        cfg.algorithm = FFTop::CPUAlgorithm::Recursive;
    } else if (spec.traversal == "dft") {
        cfg.algorithm = FFTop::CPUAlgorithm::DFT;
    } else if (spec.traversal == "four-step") {  // iterative with the planner's split
        cfg.algorithm = FFTop::CPUAlgorithm::Iterative;
        cfg.split = FFTop::CPUPlanner(sys, params).split_for(spec.size);
    } else {
        err = "unknown CPU traversal '" + spec.traversal + "'";
    }
    plan_spec.config = cfg;
    return err.empty();
}

int choose_repeats(double one_ms) {
    if (!(one_ms > 0.0) || !std::isfinite(one_ms)) return MAX_REPEATS;
    return std::clamp(static_cast<int>(std::lround(TARGET_MS / one_ms)), MIN_REPEATS, MAX_REPEATS);
}

template <typename Fn>
double time_ms(const BenchSpec& spec, Fn&& fn) {
    fn();
    const auto t_one0 = Clock::now();
    fn();
    const double one_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_one0).count();
    const int repeats = choose_repeats(one_ms);

    const auto t0 = Clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / repeats;

    std::cerr << spec.backend << ' ' << spec.traversal << " r" << spec.radix << ' ' << spec.simd
              << " t" << spec.threads << " N=" << spec.size << " repeats=" << repeats << ' ' << ms
              << " ms\n";
    return ms;
}

void print_row(const BenchSpec& spec, double ms, std::uint64_t global_mem_bytes) {
    const double bandwidth_gbps = (global_mem_bytes == 0 || !(ms > 0.0))
                                      ? 0.0
                                      : static_cast<double>(global_mem_bytes) / (ms * 1.0e6);
    const double throughput_gsamples =
        (spec.size == 0 || !(ms > 0.0)) ? 0.0 : static_cast<double>(spec.size) / (ms * 1.0e6);

    std::cout << spec.backend << ',' << spec.traversal << ',' << spec.radix << ',' << spec.simd
              << ',' << spec.execution << ',' << spec.threads << ',' << spec.size << ',' << ms
              << ',' << global_mem_bytes << ',' << bandwidth_gbps << ',' << throughput_gsamples
              << '\n';
}

#if defined(FFTOP_ENABLE_CUDA)
double time_gpu_ms(const BenchSpec& spec, const FFTop::GPUPlan& gpu) {
    using FFTop::Complex;
    const std::size_t n = spec.size;
    const std::size_t bytes = n * sizeof(Complex);
    FFTop::GPU::DeviceBuffer d_src(n);
    FFTop::GPU::DeviceBuffer d_a(n);
    FFTop::GPU::DeviceBuffer d_b(n);  // scratch

    FFTop::Buffer input(n, {1, 0});
    FFTop::GPU::check_cuda(cudaMemcpy(d_src.ptr, input.data(), bytes, cudaMemcpyHostToDevice),
                           "bench H2D");
    auto restore = [&] {
        FFTop::GPU::check_cuda(cudaMemcpy(d_a.ptr, d_src.ptr, bytes, cudaMemcpyDeviceToDevice),
                               "bench restore");
    };
    auto run = [&] { gpu.run_device(d_a.ptr, d_b.ptr, FFTop::Direction::Forward); };

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
    std::cerr << spec.backend << ' ' << spec.traversal << " r" << spec.radix << ' ' << spec.simd
              << " t" << spec.threads << " N=" << spec.size << " repeats=" << repeats << ' ' << ms
              << " ms\n";
    return ms;
}
#endif

bool run_spec(const BenchSpec& spec, bool dry, std::string& err) {
    if (dry) {
        std::cerr << spec.backend << ',' << spec.traversal << ',' << spec.radix << ',' << spec.simd
                  << ',' << spec.execution << ',' << spec.threads << ',' << spec.size << '\n';
        return true;
    }

    if (spec.backend == "naive-dft") {
        FFTop::Buffer input(spec.size, {1, 0});
        const double ms = time_ms(spec, [&] { (void)FFTop::Ref::dft(input); });
        print_row(spec, ms, 0);
        return true;
    }

    if (spec.backend != "CPU" && spec.backend != "GPU" && spec.backend != "auto") {
        err = "unknown backend '" + spec.backend + "'";
        return false;
    }

    FFTop::PlanSpec plan_spec;
    std::unique_ptr<FFTop::Plan> plan;
    try {
        if (!parse_spec(plan_spec, spec, err)) return false;
        plan = FFTop::build_plan(plan_spec);  // throws for configs that can't run N
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    if (spec.backend == "auto") std::cerr << "auto -> " << FFTop::describe(plan_spec) << '\n';

#if defined(FFTOP_ENABLE_CUDA)
    if (plan_spec.target() == FFTop::HardwareTarget::GPU) {
        const auto* gpu = dynamic_cast<const FFTop::GPUPlan*>(plan.get());
        if (!gpu) {
            err = "GPU backend unavailable";
            return false;
        }
        const double ms = time_gpu_ms(spec, *gpu);
        print_row(spec, ms, gpu->estimated_global_bytes());
        return true;
    }
#endif

    FFTop::Buffer input(spec.size, {1, 0});
    FFTop::Buffer output;
    const double ms =
        time_ms(spec, [&] { plan->execute(output, input, FFTop::Direction::Forward); });
    print_row(spec, ms, 0);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool dry = false;
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
        std::cerr << "repeats: target " << TARGET_MS << " ms, clamp [" << MIN_REPEATS << ", "
                  << MAX_REPEATS << "]\n";
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
        if (!parse_spec(spec, line, err)) {
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
