#include "fftop/backend/cpu/cpu.h"

#include "fftop/backend/cpu/simd/kernels.h"
#include "fftop/math/fft_math.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace FFTop {
namespace {
constexpr std::size_t TRANSPOSE_BLOCK = 8;   // tiled transpose for four-step
constexpr std::size_t STACK_SIZE      = 64;  // keep DFT outputs on stack for tiny N

/// @brief Transpose src into dst
/// @param dst transposed buffer with size cols x rows (row-major)
/// @param src original buffer with size rows x cols (row-major)
/// @param rows number of rows of original buffer
/// @param cols number of cols of original buffer
/// @param threads number of threads working on transpose
void transpose(Complex* dst, const Complex* src, std::size_t rows, std::size_t cols, int threads) {
    using CPU::omp_index;
    const std::size_t row_blocks = Math::ceil_div(rows, TRANSPOSE_BLOCK);
    const bool par               = CPU::use_parallel(threads, row_blocks);

    FFTOP_OMP_FOR_IF(par, threads)
    for (omp_index rb = 0; rb < static_cast<omp_index>(row_blocks); ++rb) {
        const std::size_t i0 = static_cast<std::size_t>(rb) * TRANSPOSE_BLOCK;
        const std::size_t i1 = std::min(i0 + TRANSPOSE_BLOCK, rows);
        for (std::size_t j0 = 0; j0 < cols; j0 += TRANSPOSE_BLOCK) {
            const std::size_t j1 = std::min(j0 + TRANSPOSE_BLOCK, cols);
            for (std::size_t i = i0; i < i1; ++i)
                for (std::size_t j = j0; j < j1; ++j) dst[j * rows + i] = src[i * cols + j];
        }
    }
}

std::unique_ptr<CPU::ITraversalStrategy> make_traversal(CPUAlgorithm algorithm) {
    if (algorithm == CPUAlgorithm::Recursive)
        return std::make_unique<CPU::RecursiveTraversalStrategy>();
    return std::make_unique<CPU::IterativeTraversalStrategy>();
}
}  // namespace

namespace CPU {
std::shared_ptr<const Twiddles> twiddles(std::size_t n, std::size_t radix) {
    static TableCache<std::pair<std::size_t, std::size_t>, Twiddles> cache;
    return cache.get({n, radix},
                     [&] { return std::make_shared<const Twiddles>(make_twiddles(n, radix)); });
}
}  // namespace CPU

void CPUPlan::check_input(const Buffer& input) const {
    if (input.size() != n_)
        throw std::invalid_argument(name_ + ": built for N=" + std::to_string(n_) + ", got " +
                                    std::to_string(input.size()));
}

void CPUPlan::execute(Buffer& output, const Buffer& input, Direction dir) const {
    check_input(input);
    output = input;
    run(output.data(), dir);
}

// ====================== FFT =========================
CPUFFTPlan::CPUFFTPlan(std::size_t n, const CPUConfig& cfg, std::string name)
    : CPUPlan(n, std::move(name))
    , traversal_(make_traversal(cfg.algorithm))
    , threads_(cfg.threads) {
    if (cfg.split == 0) {
        rows1_ = make_sub_fft(n, cfg);
        return;
    }
    rows1_ = make_sub_fft(n / cfg.split, cfg);  // first log2(N2) stages
    rows2_ = make_sub_fft(cfg.split, cfg);
    assert(rows1_.twiddles->n * rows2_.twiddles->n == n);

    log_m_              = (Math::log2_floor(n) + 1) / 2;
    const std::size_t m = std::size_t(1) << log_m_;
    lo_twiddles_.resize(m);
    for (std::size_t j = 0; j < m; ++j) lo_twiddles_[j] = root_of_unity(j, n);
    hi_twiddles_.resize(Math::ceil_div(n, m));
    for (std::size_t i = 0; i < hi_twiddles_.size(); ++i) hi_twiddles_[i] = root_of_unity(i * m, n);
}

CPUFFTPlan::SubFFT CPUFFTPlan::make_sub_fft(std::size_t n, const CPUConfig& cfg) {
    SubFFT sub;
    if (cfg.radix == Radix::Radix4 && Math::is_power_of(n, 4))
        sub.butterfly = std::make_unique<CPU::Radix4>(cfg.simd);
    else
        sub.butterfly = std::make_unique<CPU::Radix2>(cfg.simd);
    sub.twiddles = CPU::twiddles(n, sub.butterfly->radix());
    return sub;
}

void CPUFFTPlan::run_sub_fft(const SubFFT& sub, Complex* data, Direction dir, int threads) const {
    traversal_->run(data, sub.twiddles->n, dir, *sub.butterfly, threads, *sub.twiddles);
}

Complex CPUFFTPlan::twiddle(std::size_t e, Direction dir) const {
    const Complex w =
        CPU::cmul(hi_twiddles_[e >> log_m_], lo_twiddles_[e & (lo_twiddles_.size() - 1)]);
    return dir == Direction::Forward ? w : Complex{w.real(), -w.imag()};
}

void CPUFFTPlan::four_step(Buffer& out, const Buffer& in, Direction dir) const {
    using CPU::omp_index;
    const std::size_t n1 = rows1_.twiddles->n;
    const std::size_t n2 = rows2_.twiddles->n;

    Buffer tmp(size());
    out.resize(size());

    Complex* T = tmp.data();
    Complex* Y = out.data();

    // 1. Transpose: x (N1 x N2) -> t (N2 x N1)
    transpose(T, in.data(), n1, n2, threads_);

    // 2. Run N1-length subFFT on N2 rows
    FFTOP_OMP_FOR_IF(CPU::use_parallel(threads_, n2), threads_)
    for (omp_index r = 0; r < static_cast<omp_index>(n2); ++r) {
        Complex* row = T + static_cast<std::size_t>(r) * n1;
        run_sub_fft(rows1_, row, dir, 1);

        // Apply twiddle
        std::size_t e = 0;
        for (std::size_t k1 = 0; k1 < n1; ++k1, e += static_cast<std::size_t>(r))
            row[k1] = CPU::cmul(row[k1], twiddle(e, dir));
    }

    // 3. Transpose: t (N2 x N1) -> y (N1 x N2)
    transpose(Y, T, n2, n1, threads_);

    // 4. Run N2-length subFFT on N1 rows
    FFTOP_OMP_FOR_IF(CPU::use_parallel(threads_, n1), threads_)
    for (omp_index r = 0; r < static_cast<omp_index>(n1); ++r) {
        run_sub_fft(rows2_, Y + static_cast<std::size_t>(r) * n2, dir, 1);
    }

    // 5. y (N1 x N2) -> X (N2 x N1), lands in tmp; swap it into out, no copy
    transpose(T, Y, n1, n2, threads_);
    out.swap(tmp);
}

// No-split FFT
void CPUFFTPlan::run(Complex* data, Direction dir) const {
    run_sub_fft(rows1_, data, dir, threads_);
}

void CPUFFTPlan::execute(Buffer& output, const Buffer& input, Direction dir) const {
    check_input(input);
    if (split())  // four-step split
        return four_step(output, input, dir);
    // else do 1D FFT
    output = input;
    run(output.data(), dir);
}

// ====================== DFT =========================

CPUDFTPlan::CPUDFTPlan(std::size_t n, std::string name)
    : CPUPlan(n, std::move(name))
    , w_(n) {
    for (std::size_t k = 0; k < n; ++k) w_[k] = root_of_unity(k, n);
}

void CPUDFTPlan::run(Complex* data, Direction dir) const {
    const std::size_t n = size();
    if (n <= 1) return;

    std::array<Complex, STACK_SIZE> stack_out;
    std::vector<Complex> heap_out;
    Complex* out = stack_out.data();
    if (n > STACK_SIZE) {
        heap_out.resize(n);
        out = heap_out.data();
    }

    const Real sign = dir == Direction::Forward ? Real(1) : Real(-1);  // inverse: conjugate
    for (std::size_t k = 0; k < n; ++k) {
        Real re = 0, im = 0;
        std::size_t idx = 0;  // (k * t) mod n
        for (std::size_t t = 0; t < n; ++t) {
            const Real wr = w_[idx].real();
            const Real wi = sign * w_[idx].imag();
            const Real xr = data[t].real();
            const Real xi = data[t].imag();
            re += wr * xr - wi * xi;
            im += wr * xi + wi * xr;
            idx += k;
            if (idx >= n) idx -= n;
        }
        out[k] = {re, im};
    }
    std::copy(out, out + n, data);
}

std::unique_ptr<Plan> make_cpu_plan(std::size_t n, const CPUConfig& cfg) {
    validate(cfg, n);
    std::string name = "CPU/" + to_string(cfg);
    if (cfg.algorithm == CPUAlgorithm::DFT) return std::make_unique<CPUDFTPlan>(n, std::move(name));
    return std::make_unique<CPUFFTPlan>(n, cfg, std::move(name));
}
}  // namespace FFTop
