# FFT-op: An FFT library with hardware-specific optimizations

FFT-op provides a common FFT interface with implementations optimized for different hardware. The goal is to explore how algorithm choice, parallelism, vectorization, and memory access affect FFT performance across CPUs and GPUs.

## Usage

```bash
cmake -B build && cmake --build build
```

CUDA is enabled automatically when a CUDA compiler is available. Pass `-DFFTOP_ENABLE_CUDA=OFF` to force a CPU-only build.

The main API is `FFTop::fft()`:

```cpp
FFTop::fft(x);                                          
// Use Backend::Auto to automatically resolves
FFTop::fft(x, {Backend::CPU, Direction::Forward});
FFTop::fft(x, {Backend::GPU, Direction::Inverse});
```

Inverse FFTs are unnormalized (no `/N`).

Run tests and benchmarks with:

```bash
ctest --test-dir build
./build/bench/fftop_bench
```

## Implementations

FFT-op detects available CPU features, thread count, SIMD support (SSE2 / AVX2 / AVX-512 / NEON), OpenMP, and CUDA devices at runtime. It uses this information to select kernels and resolve `Backend::Auto`.

**CPU**

* Radix-2 and radix-4
* Scalar, AVX2, and NEON kernels
* Optional OpenMP parallelism
* Iterative and recursive implementations

**GPU**

* CUDA radix-2 and radix-4
* Stockham radix-2
* Naive DFT for reference

FFT plans are cached and reused across repeated transforms with the same configuration.
