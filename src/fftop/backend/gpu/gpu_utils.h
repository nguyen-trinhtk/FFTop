#pragma once

// Internal utilities shared across GPU backends.
// Safe to include from CUDA-enabled .cu and .cpp files.

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace FFTop::GPU {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

inline unsigned blocks_for(unsigned n, unsigned block = 256) {
    return n == 0 ? 0 : (n + block - 1) / block;
}

inline unsigned log2_floor(unsigned n) {
    unsigned e = 0;
    while (n > 1) {
        n >>= 1;
        ++e;
    }
    return e;
}

struct DeviceBuf {
    double2*    ptr = nullptr;
    std::size_t n   = 0;

    DeviceBuf() = default;

    explicit DeviceBuf(std::size_t count) : n(count) {
        if (count == 0) return;
        check_cuda(cudaMalloc(&ptr, count * sizeof(double2)), "cudaMalloc");
    }

    ~DeviceBuf() {
        if (ptr) cudaFree(ptr);
    }

    DeviceBuf(const DeviceBuf&)            = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;

    DeviceBuf(DeviceBuf&& o) noexcept : ptr(o.ptr), n(o.n) {
        o.ptr = nullptr;
        o.n   = 0;
    }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept {
        if (this != &o) {
            if (ptr) cudaFree(ptr);
            ptr   = o.ptr;
            n     = o.n;
            o.ptr = nullptr;
            o.n   = 0;
        }
        return *this;
    }
};

}  // namespace FFTop::GPU
