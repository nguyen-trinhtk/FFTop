# Optimize FFT implementation

This project's primary goal is to discover & benchmark different hardware-efficient implementations of the Fast Fourier Transform algorithm, using a mix-and-match of techniques while considering the computing architecture beneath. Currently the focus is on CPU implementations.

Please refer to [this document](./doc/optimizations.md) to learn more about each optimization.

---

### Getting started

```bash
make test    # run correctness tests
make bench   # run benchmarks (requires FFTW3)
make clean   # remove build artifacts
```

Compiler flags (override with `make CXX=g++` on Linux):

```
g++ -g -O3 -march=native -std=c++17 -Wall -Wextra -Iinclude
```

On macOS with Homebrew, the Makefile auto-detects `g++-15` when present.

---

### Benchmarking

Steady-state results at `N = 16777216` (see [full log](./bench/results/bench.md)):

| Variant | Avg ms |
|---------|-------:|
| Radix-2 FFT | 5628 |
| Radix-4 FFT | 3662 |
| Iterative In-Place FFT | 1038 |
| SIMD Iterative In-Place FFT | 938 |
| OpenMP+SIMD Iterative In-Place FFT | 881 |
| Four-Step FFT | 1014 |
| Parallel Four-Step FFT | 301 |
| FFTW3 (steady) | 387 |

Benchmarking is done on a Silicon M3 laptop (4×4.05 GHz + 4×2.75 GHz cores, ~100 GB/s memory bandwidth).

#### TODO: GPU benchmarking

---

### Project layout

```
include/fft/
  core/         headers: types, bit ops, complex utilities
  ref/          reference DFT declaration
  cpu/          public variant declarations + detail/ internal headers
src/fft/
  ref/          reference DFT implementation
  cpu/          variant implementations + detail/ building blocks
test/           correctness harness
bench/          benchmark harness
doc/            theory and optimization notes
```

Headers live under `include/`; implementations under `src/`. The Makefile passes `-Iinclude` so includes use paths like `#include "fft/cpu/iterative.h"`.

---

### Testing

All implemented variants are tested against naive DFT, which are ensured to be correct. The following tests are conducted:

- Test correctness for input sizes 2^k
- Edge cases:
    + All-zeros input
    + Impulse at 0
    + Impulse at k
- Parseval's theorem: total energy of signal in time domain equals to that of frequency domain
