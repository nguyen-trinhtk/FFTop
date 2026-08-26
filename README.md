# FFT-op: An FFT library for CPUs and GPUs

A small C++ library for complex FFTs. The CPU path is what works today. GPU
support is a placeholder so there is a place to put it later.

## Status

Pass in a power-of-two number of samples and you get a forward or inverse
transform on the CPU. Longer transforms that also happen to be a power of four
use a faster kernel automatically; everything else uses the simpler one.

The inverse does not scale the result back down. If you transform and then
invert, you get the original times the length, not the original.

GPU is not implemented. Asking for it will not give you an FFT.

## Build

CMake 3.20 and a C++17 compiler:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Tests and a small benchmark come along for the ride. Parallelism is used when
the machine has it; otherwise the build stays single-threaded. On a Mac, that
usually means installing Homebrew’s OpenMP package.

## Usage

Headers are in `include/`. Link against `fftop::fftop`.

```cpp
#include "fftop/fft.h"

FFTop::Buffer x = /* N complex samples, N a power of two */;
FFTop::Buffer y = FFTop::fft(x);   // forward, on the CPU

FFTop::Buffer z = FFTop::fft(y, {FFTop::Backend::CPU,
                                 FFTop::Direction::Inverse});
```

The input is left alone; you get a new buffer back. The only knobs you pass in
are which device and which direction. How the transform is factored is decided
for you.

## CPU backend

Vector instructions, the butterflies, the order they run in, and whether they
run in parallel are separate pieces. You should be able to change one without
rewriting the others. The planner looks at the length and picks the faster
kernel when it can.

## Limits

Power-of-two lengths only, and only double-precision complex numbers. Lengths
that are not a power of two will fail a debug assert rather than quietly
return garbage.

The inverse is unnormalized, as above. The CPU path still computes rotation
factors on the fly, so it is correct but not as fast as it will get. And the
GPU path does nothing.

## Benchmarks

```bash
cmake --build build --target fftop_bench
./build/bench/fftop_bench
```

That prints time-per-transform for a few sizes, comparing the two CPU kernels
on equal footing. Use a Release build or the numbers will look worse than they
are.

## Roadmap

See [`doc/TODO.md`](doc/TODO.md). In short: any length, not just powers of two;
an in-place call so you do not need a second buffer; and a GPU kernel that
actually runs.
