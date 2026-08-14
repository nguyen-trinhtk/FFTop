# Optimize FFT implementation

This project's primary goal is to discover & benchmark different hardware-efficient implementations of the Fast Fourier Transform algorithm, using a mix-and-match of techniques while considering the computing architecture beneath.

Please refer to [this document](./doc/optimizations.md) to learn more about each optimization.



---

### Getting started

```bash
# CPU (local)
make test-cpu    # or: make test
make bench-cpu   # or: make bench  (requires FFTW3)
make clean

# GPU (CUDA machine / Colab T4)
make test-gpu CUDA_ARCH=sm_75
make bench-gpu CUDA_ARCH=sm_75
```

Compiler flags (override with `make CXX=g++` on Linux):

```
g++ -g -O3 -march=native -std=c++17 -Wall -Wextra -Iinclude
```

On macOS with Homebrew, the Makefile auto-detects `g++-15` when present.

---

### Benchmarking

Steady-state CPU results at `N = 16777216` (see [full log](./bench/results/bench.md)):

| Variant | Avg ms |
|---------|-------:|
| Radix-2 FFT | 5628 |
| Radix-4 FFT | 3662 |
| Iterative In-Place FFT | 1038 |
| OpenMP Iterative In-Place FFT | 881 |
| Four-Step FFT | 1014 |
| Parallel Four-Step FFT | 301 |
| FFTW3 (steady) | 387 |

Benchmarking is done on a Silicon M3 laptop (4×4.05 GHz + 4×2.75 GHz cores, ~100 GB/s memory bandwidth).

#### GPU benchmarking

Compare before/after on a CUDA machine (Colab T4):

```bash
make test-gpu bench-gpu CUDA_ARCH=sm_75
```

| Variant | N = 1024 | N = 4096 |
|---------|---------:|---------:|
| GPU Naive (before) | 0.438 | 0.739 |
| GPU Optimized (warp/shared locality) | 0.244 | 0.315 |

(Avg ms. Optimized fuses early stages via warp `__shfl_*` + `__shared__` tiles.)

Benchmarking is done on NVIDA Tesla T4 GPU on Google Colab. Unfortunately, I do not own a dedicated GPU for running these.

---

### Project layout

```
include/fft/
  core/         shared types, bit ops, complex utilities
  ref/          reference DFT (correctness oracle)
  cpu/          CPU public APIs + detail/
  gpu/          GPU public APIs + detail/
src/fft/
  ref/          reference DFT implementation
  cpu/          CPU variants + detail/ building blocks
  gpu/          CUDA kernels (.cu) + detail/
test/
  cpu_implementations.cpp   # registered CPU variants
  gpu_implementations.cpp   # registered GPU variants
bench/
  cpu_implementations.cpp   # CPU + FFTW bench registry
  gpu_implementations.cpp   # GPU bench registry
doc/
```

Headers live under `include/`; implementations under `src/`. The Makefile passes `-Iinclude` so includes use paths like `#include "fft/cpu/iterative.h"`.

CPU and GPU builds are separate binaries — `test-gpu` / `bench-gpu` do not link CPU FFT variants (DFT is linked only as the test oracle).

---

### Testing

All implemented variants are tested against naive DFT, which are ensured to be correct. The following tests are conducted:

- Test correctness for input sizes 2^k
- Edge cases:
    + All-zeros input
    + Impulse at 0
    + Impulse at k
- Parseval's theorem: total energy of signal in time domain equals to that of frequency domain
