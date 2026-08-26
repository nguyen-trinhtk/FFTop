#pragma once

// Internal utilities shared across GPU backends.
// Only include this from .cu translation units.

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace FFTop::GPU {

inline void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

struct DeviceBuf {
    double2* ptr = nullptr;

    explicit DeviceBuf(std::size_t n) {
        check_cuda(cudaMalloc(&ptr, n * sizeof(double2)), "cudaMalloc");
    }

    ~DeviceBuf() { cudaFree(ptr); }

    DeviceBuf(const DeviceBuf&)            = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
};

}  // namespace FFTop::GPU
