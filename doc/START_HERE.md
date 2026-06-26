
# Start Here

Personal (git-ignored) notes during implementation

### Roadmap

DFT -> Radix-2 FFT -> Radix-4 FFT -> Iterative in-place & bit reversal -> Four step Bailey FFT -> SIMD vectorized (AVX2/NEON) FFT -> OpenMP parallel FFT -> GPU-CUDA / OpenCL shared mem, warp butterfly -> cuFFT -> Warp-shuffle micro-FFT (register, no share mem)

### Repo struct

```
fft/
├── ref/                  # Stage 1: your oracle, never modified
│   └── dft.cpp
│
├── core/                 # Shared primitives all impls use
│   ├── twiddle.hpp       # Precomputed twiddle table
│   ├── bitrev.hpp        # Bit-reversal permutation
│   └── types.hpp         # Complex type, precision typedef
│
├── cpu/
│   ├── radix2.cpp        # Each variant is a self-contained file
│   ├── radix4.cpp
│   ├── split_radix.cpp
│   └── simd/
│       └── avx2.cpp
│
├── gpu/
│   ├── naive.cu
│   └── warp_shuffle.cu
│
├── fpga/
│   └── hls/
│
├── bench/                # Benchmarks are first-class, not afterthoughts
│   ├── runner.cpp        # Unified benchmark harness
│   └── results/          # Committed JSON/CSV — you want history here
│
└── test/
    └── correctness.cpp   # Runs ALL variants against ref/dft.cpp
```


### Test 

New test: create a new test_*.cpp file and add one TestRegistrar

New implementation: add one ImplementationRegistrar in implementations.cpp (line 1) or another implementation file

### Memcheck
The most useful lines to look at are:

- maximum resident set size
- peak memory footprint
- 0 leaks for 0 total leaked bytes

### Twiddle LUT
What to watch for
The twiddle multiplication wn1 *= w1 accumulates floating-point error over long loops. For large N, precompute a twiddle table instead:

```cpp
// Precompute once per stage
std::vector<cd> twiddles(len/4);
for (int j = 0; j < len/4; j++)
    twiddles[j] = std::polar(1.0, -2.0 * PI * j / len);
```

### Mixed radix (ignored for now)
Mixed-radix fallback: if log₂(N) is odd (N = 32, 128, 512...), you can't decompose purely into radix-4 stages. The standard fix is to do one radix-2 stage first, then all remaining stages as radix-4. FFTW does exactly this.