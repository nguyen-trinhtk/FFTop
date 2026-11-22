#pragma once

#include "fftop/plan/plan.h"
#include "fftop/backend/cpu/butterfly.h"
#include "fftop/backend/cpu/parallel.h"
#include "fftop/backend/cpu/traversal.h"
#include "fftop/math/twiddle.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace FFTop {

namespace CPU {
// Host twiddle table for (n, radix)
std::shared_ptr<const Twiddles> twiddles(std::size_t n, std::size_t radix);
}  // namespace CPU

// Abstract base for CPU plans: in-place FFT on N contiguous elements
class CPUPlan : public Plan {
public:
    CPUPlan(std::size_t n, std::string name)
        : n_(n)
        , name_(std::move(name)) {}

    std::string name() const override { return name_; }
    std::size_t size() const override { return n_; }

    /// @brief Copies input to output then run in-place FFT on output via `run()`
    /// @param output Output data buffer
    /// @param input Input data buffer
    /// @param dir Forward or inverse
    void execute(Buffer& output, const Buffer& input, Direction dir) const override;

protected:
    virtual void run(Complex* data, Direction dir) const = 0;
    void check_input(const Buffer& input) const;
private:
    std::size_t n_;
    std::string name_;
};

/// @brief FFT butterfly network (recursive or iterative), possibly with four-step transpose
class CPUFFTPlan final : public CPUPlan {
public:
    CPUFFTPlan(std::size_t n, const CPUConfig& cfg, std::string name);
    void execute(Buffer& output, const Buffer& input, Direction dir) const override;

private:
    void run(Complex* data, Direction dir) const override;

    // A 1D FFT of length twiddles->n
    struct SubFFT {
        std::unique_ptr<CPU::IRadixR> butterfly;
        std::shared_ptr<const Twiddles> twiddles;  // cached, shared
    };
    static SubFFT make_sub_fft(std::size_t n, const CPUConfig& cfg);

    bool split() const { return rows2_.twiddles != nullptr; }  // four-step split or not
    void run_sub_fft(const SubFFT& sub, Complex* data, Direction dir, int threads) const;

    // in -> out (resized to N). in and out may be the same buffer.
    void four_step(Buffer& out, const Buffer& in, Direction dir) const;
    Complex twiddle(std::size_t e, Direction dir) const;  // W_N^e

    std::unique_ptr<CPU::ITraversalStrategy> traversal_;
    int threads_;
    SubFFT rows1_;  // no split: N1=N, split: rows of length N1
    SubFFT rows2_;  // N2 after-transpose rows (split only)

    // Split only
    unsigned log_m_ = 0;
    std::vector<Complex> lo_twiddles_;  // twiddles for stages before transpose
    std::vector<Complex> hi_twiddles_;  // twiddles for stages after transpose
};

// Direct O(N^2) DFT from twiddle table, suitable for tiny N
class CPUDFTPlan final : public CPUPlan {
public:
    CPUDFTPlan(std::size_t n, std::string name);
private:
    void run(Complex* data, Direction dir) const override;
    std::vector<Complex> w_;  // where w_[k] = W_N^k
};

std::unique_ptr<Plan> make_cpu_plan(std::size_t n, const CPUConfig& cfg);
}  // namespace FFTop
