#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace FFTGpu {
namespace detail {

#define FFT_CUDA_CHECK(call)                                                   \
    do {                                                                       \
        const cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(                                          \
                std::string("CUDA error at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));    \
        }                                                                      \
    } while (0)

__device__ __forceinline__ int bit_reverse_device(int i, int bits) {
    int r = 0;
    for (int b = 0; b < bits; ++b) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ double2 cadd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ double2 csub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

inline int div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

inline int choose_four_step_n1(int n) {
    int log2n = 0;
    for (int t = n; t > 1; t >>= 1) {
        ++log2n;
    }
    return 1 << (log2n / 2);
}

// Owning device buffer for double2 (matches std::complex<double> layout).
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        allocate(count);
    }

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        if (count == 0) {
            return;
        }
        FFT_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(double2)));
        count_ = count;
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    double2* get() const {
        return ptr_;
    }

    std::size_t size() const {
        return count_;
    }

    std::size_t bytes() const {
        return count_ * sizeof(double2);
    }

    void upload(const void* host) {
        FFT_CUDA_CHECK(cudaMemcpy(ptr_, host, bytes(), cudaMemcpyHostToDevice));
    }

    void download(void* host) const {
        FFT_CUDA_CHECK(cudaMemcpy(host, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

    void copy_from(const DeviceBuffer& src) {
        FFT_CUDA_CHECK(
            cudaMemcpy(ptr_, src.ptr_, bytes(), cudaMemcpyDeviceToDevice));
    }

private:
    double2* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace detail
}  // namespace FFTGpu
