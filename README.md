# FFT-op: An FFT library with hardware-specific optimizations

FFT-op is a C++ FFT library for CPUs and GPUs. There are different implementations for different hardware, with optimizations in the FFT algorithm, parallelism/vectorization, and memory layout.

The library picks a backend and radix based on FFT size and available hardware. Type-based tuning is planned.

For a detailed optimization walkthrough, see the [worklog](./doc/Worklog.md).

## Software support

FFT-op currently detects:

**CPU**
- Thread count
- OpenMP
- SIMD: AVX2, AVX-512, NEON

**GPU**
- CUDA (NVIDIA GPUs only)

CUDA and OpenMP are configurable through the root `CMakeLists.txt`. Thread count and SIMD support are detected at runtime.

This information is used to resolve `HardwareTarget::Auto` and pick the best available kernel. There are kernels for AVX-512, AVX2, and NEON; x86 CPUs without AVX2 fall back to scalar.

FFT execution plans are cached and reused for repeated transforms with the same configuration.
