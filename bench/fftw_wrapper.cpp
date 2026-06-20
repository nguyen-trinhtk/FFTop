#include "fftw_wrapper.h"

#include <fftw3.h>

namespace {

struct FftwCache {
    int n = 0;
    fftw_complex* in = nullptr;
    fftw_complex* out = nullptr;
    fftw_plan plan = nullptr;

    ~FftwCache() {
        reset();
    }

    void reset() {
        if (plan != nullptr) {
            fftw_destroy_plan(plan);
            plan = nullptr;
        }
        if (in != nullptr) {
            fftw_free(in);
            in = nullptr;
        }
        if (out != nullptr) {
            fftw_free(out);
            out = nullptr;
        }
        n = 0;
    }

    void ensure(int size) {
        if (n == size) {
            return;
        }
        reset();
        n = size;
        in = fftw_alloc_complex(size);
        out = fftw_alloc_complex(size);
        plan = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_MEASURE);
    }
};

FftwCache& fftw_cache() {
    static thread_local FftwCache cache;
    return cache;
}

void copy_in(const std::vector<FFTCore::Complex>& input, fftw_complex* buffer) {
    for (std::size_t i = 0; i < input.size(); ++i) {
        buffer[i][0] = input[i].real();
        buffer[i][1] = input[i].imag();
    }
}

void copy_out(const fftw_complex* buffer, std::size_t n, std::vector<FFTCore::Complex>& output) {
    output.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = FFTCore::Complex(buffer[i][0], buffer[i][1]);
    }
}

}  // namespace

void fftw_fft_cold(const std::vector<FFTCore::Complex>& input,
                   std::vector<FFTCore::Complex>& output) {
    const int n = static_cast<int>(input.size());
    fftw_complex* in = fftw_alloc_complex(n);
    fftw_complex* out = fftw_alloc_complex(n);
    copy_in(input, in);

    fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    copy_out(out, input.size(), output);

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

void fftw_fft_steady(const std::vector<FFTCore::Complex>& input,
                     std::vector<FFTCore::Complex>& output) {
    const int n = static_cast<int>(input.size());
    FftwCache& cache = fftw_cache();
    cache.ensure(n);
    copy_in(input, cache.in);
    fftw_execute(cache.plan);
    copy_out(cache.out, input.size(), output);
}
