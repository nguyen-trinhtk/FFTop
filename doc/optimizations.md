# Progressive optimization walkthrough

### I. Naive DFT

This is the foundational implementation of Fourier Transform, with time complexity of $O(n^2)$. Without changing the time complexity yet, this can be optimized with better exploit of parallelism and usage of cache; however we would not focus on that due to a higher impact algorithmic update in FFT.

### II. Cooley-Tukey FFT (radix-2)
This recursive algorithm reduces asymptotic complexity from $O(n^2)$ to $O(n \log{n})$, which made room for substantial performance gains especially over large input. However there will be better implementation that utilizes the hardware more efficiently.

### III. Radix-4


### TODO: Remarks on FFTW planner for radix selection
- Director (planner) vs executor pattern
- Adapt to hardware (treat as parameter)

---

## Module architecture

Code is organized in three layers, with headers under `include/fft/` and sources under `src/fft/`:

| Layer | Headers | Sources | Role |
|-------|---------|---------|------|
| Core | `include/fft/core/` | — | Types, bit-reversal, `log2_floor`, small math helpers |
| Building blocks | `include/fft/cpu/detail/` | `src/fft/cpu/detail/` | Reusable algorithms — not registered as benchmark variants |
| Public variants | `include/fft/cpu/*.h` | `src/fft/cpu/*.cpp` | Thin wrappers exposed to test/bench registries |

### `fft/cpu/detail/` modules

| Module | Responsibility |
|--------|----------------|
| `iterative_radix2` | Scalar in-place radix-2 FFT (on-the-fly twiddles) |
| `simd_radix2` | SIMD in-place radix-2 FFT via `simd-butterfly.h`; optional OpenMP across groups |
| `matrix_ops` | Four-step helpers: `choose_four_step_n1`, blocked transpose, twiddle multiply |
| `four_step` | Full four-step decomposition driven by `FourStepPolicy` |

`FourStepPolicy` selects serial vs parallel transpose/twiddles, SIMD vs scalar row FFTs, and the small-N fallback threshold. New variants (e.g. serial SIMD four-step) are a one-line policy change, not a fork of the algorithm.

Intentional forks kept separate for benchmarking: recursive radix-2/4 (`radix-2.cpp`, `radix-4.cpp`) vs iterative paths; scalar on-the-fly twiddles vs precomputed SIMD twiddles.

