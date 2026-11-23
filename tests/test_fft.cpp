#include "fftop/fft.h"
#include "fftop/system.h"
#include "support/check.h"
#include "support/cpu.h"

#include <gtest/gtest.h>

#include <stdexcept>

#if defined(FFTOP_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

using namespace FFTop;
using namespace FFTop::Test;

TEST(Fft, MatchesDft) {
    for (HardwareTarget target : {HardwareTarget::CPU, HardwareTarget::Auto}) {
        auto transform = [&](const Buffer& x, Direction dir) { return fft(x, {target, dir}); };
        for_each_input(POWERS_OF_TWO, [&](const Buffer& x) { expect_matches_dft(transform, x); });
    }
}

// Whatever the planner picks for a big size (four-step on a small-L2 host)
// must agree with a plain iterative FFT.
TEST(Fft, LargeSizeAgreesWithIterative) {
    const std::size_t n = std::size_t(1) << 20;
    const auto input = mixed(n);
    auto iterative = build(n, cpu_config(CPUAlgorithm::Iterative, Radix::Radix4));
    for (Direction dir : {Direction::Forward, Direction::Inverse})
        expect_close(fft(input, {HardwareTarget::CPU, dir}), run(*iterative, input, dir));
}

TEST(Fft, InverseUndoesForwardTimesN) {
    const std::size_t n = 4096;
    const auto x = mixed(n);
    Buffer back = fft(fft(x), {HardwareTarget::Auto, Direction::Inverse});
    for (auto& z : back) z /= Real(n);
    expect_close(back, x);
}

TEST(Fft, DeviceApiNeedsCuda) {
    if (system_config().cuda) GTEST_SKIP() << "CUDA device present";
    Complex dummy[16];
    EXPECT_THROW(fft_device(dummy, dummy, 16), std::runtime_error);
}

#if defined(FFTOP_ENABLE_CUDA)
TEST(Fft, DeviceApiMatchesHost) {
    if (!system_config().cuda) GTEST_SKIP() << "no CUDA device";
    for (std::size_t n : {std::size_t(16), std::size_t(4096), std::size_t(1) << 16}) {
        const auto input = mixed(n);
        const auto bytes = n * sizeof(Complex);
        Complex* d_in = nullptr;
        Complex* d_out = nullptr;
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_in), bytes), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);

        for (Direction dir : {Direction::Forward, Direction::Inverse}) {
            const Buffer want = fft(input, {HardwareTarget::CPU, dir});
            Buffer got(n);

            fft_device(d_out, d_in, n, dir);  // out of place; d_in untouched
            ASSERT_EQ(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            expect_close(got, want, "out-of-place N=" + std::to_string(n));

            ASSERT_EQ(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice), cudaSuccess);
            fft_device(d_out, d_out, n, dir);  // in place
            ASSERT_EQ(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
            expect_close(got, want, "in-place N=" + std::to_string(n));
        }
        cudaFree(d_in);
        cudaFree(d_out);
    }
}
#endif
