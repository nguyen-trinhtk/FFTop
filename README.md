# FFT-op: An FFT library with hardware-specific optimizations

FFT-op is a C++ FFT library optimized for most CPUs and NVIDIA GPUs. 


## Hardware-aware planning

FFT-op checks the hardware target and specs once per process:

- CPU: threads, physical cores, cache sizes (L1D, L2, L3), OpenMP, SIMD (AVX2, AVX-512, NEON)
- GPU: compute capability, SM count, shared memory, threads per block
- Auto: delegates to either CPU or GPU check

*(Note: `Auto` only select GPU when N is large enough to compensate for the host-device copies overhead.)*

Using those specs and the input size `N`, the planner picks the suitable configuration, which includes algorithm, radix, and parallelism parameters. Plans are cached per (size, target), so repeat sizes on the same hardware skip planning.

## Relevant docs
- [Worklog](./doc/Worklog.md): written description of optimizations

<!-- TODO: roadmap, dev manual, benchmark/profile -->

## Build

```bash
cmake -B build && cmake --build build
ctest --test-dir build    # tests
./bench/bench-sweep.sh    # benchmarks
```

CUDA and OpenMP are automatically on if included by toolchain. Pass `-DFFTOP_ENABLE_CUDA=OFF` or `-DFFTOP_ENABLE_OPENMP=OFF` to leave either out.

## Usage

```cpp
FFTop::fft(x); // Auto: CPU or GPU
FFTop::fft(x, {HardwareTarget::CPU, Direction::Forward});
FFTop::fft(x, {HardwareTarget::GPU, Direction::Inverse}); 

// For data already on the GPU (CUDA builds)
FFTop::fft_device(d_out, d_in, n, Direction::Forward);
```

For now, sizes must be powers of two. Support for arbitrary sizes (with any radix) may be added in the future.

Note that inverse FFT isn't normalized, so forward then inverse will output `N * x`.

## References

1. Cooley & Tukey. [An algorithm for the machine calculation of complex Fourier series](https://doi.org/10.1090/S0025-5718-1965-0178586-1). *Math. Comp.*, 1965. Radix-2 FFT, iterative with bit-reversed order.
2. Cochran et al. [What is the fast Fourier transform?](https://doi.org/10.1109/PROC.1967.5957) *Proc. IEEE*, 1967. Stockham autosort.
3. Bailey. [FFTs in external or hierarchical memory](https://doi.org/10.1007/BF00162341). *J. Supercomputing*, 1990. Four-step.
4. Van Loan. [Computational Frameworks for the Fast Fourier Transform](https://doi.org/10.1137/1.9781611970999). SIAM, 1992. Radix-4, Stockham and four-step in one framework.
5. Frigo & Johnson. [The design and implementation of FFTW3](https://doi.org/10.1109/JPROC.2004.840301). *Proc. IEEE*, 2005. Plans and the planner.
6. Govindaraju et al. [High performance discrete Fourier transforms on graphics processors](https://doi.org/10.1109/SC.2008.5213922). *SC '08*, 2008. GPU Stockham kernels.




