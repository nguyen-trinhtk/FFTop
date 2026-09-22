# FFT-op: An FFT library with hardware-specific optimizations

FFT-op is a C++ FFT library optimized for CPU and GPUs. The goal is to explore different optimization techniques, spanning algorithm design, concurrent execution (parallelism and vectorization), and memory layout. The library already picks a backend and radix from size and hardware; type-based tuning is planned.

For the full optimization walkthrough, please view this [worklog](./doc/Worklog.md).

## Software support

Currently, FFT-op detects available support for:

1. CPU
- Thread count
- OpenMP
- SIMD: AVX2, AVX-512, NEON

2. GPU
Currently only supports NVIDIA GPUs.
- CUDA support

CUDA and OpenMP are configurable via root CMakeLists.txt flags. Thread count and SIMD are detected at runtime.

The library uses this information to pick the best kernel the chip can run and resolve `HardwareTarget::Auto`. AVX-512, AVX2, and NEON each have kernels; x86 without AVX2 uses scalar. FFT execution plans are cached and reused across repeated transforms with the same configuration.
