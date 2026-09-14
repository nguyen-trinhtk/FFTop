# FFT-op: An FFT library with hardware-specific optimizations

FFT-op is a C++ FFT library optimized for CPU and GPUs. The goal is to explore different optimization techniques, spanning algorithm design, concurrent execution (parallelism and vectorization), and memory layout. The library already picks a backend and radix from size and hardware; type-based tuning is planned.

For the full optimization walkthrough, please view this [worklog](./doc/Worklog.md).

## Usage

```bash
cmake -B build && cmake --build build
```

CUDA is enabled automatically when a CUDA compiler is available. Pass `-DFFTOP_ENABLE_CUDA=OFF` to force a CPU-only build. OpenMP is similar (`-DFFTOP_ENABLE_OPENMP=OFF`).

The main API is `FFTop::fft()`:

```cpp
FFTop::fft(x);  // HardwareTarget::Auto
FFTop::fft(x, {HardwareTarget::CPU, Direction::Forward});
FFTop::fft(x, {HardwareTarget::GPU, Direction::Inverse});
```

If GPU is requested but CUDA is not usable, execution falls back to CPU.

Inverse FFTs are unnormalized (no `/N`). Sizes must be a power of two.

Run tests and benchmarks with:

```bash
ctest --test-dir build
./bench/bench-sweep.sh
```

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
