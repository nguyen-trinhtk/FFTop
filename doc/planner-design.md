<!-- # Planner design

How FFTop decides how to run an FFT, and how it caches that decision.

## Flow

```
fft(input, opts)  /  fft_device(d_out, d_in, n, dir)   (device data => GPU)
  PlanCache.find(size, target)
    hit  -> plan.execute(...)
    miss -> Planner::make_spec: resolve Auto, then CPUPlanner or GPUPlanner -> PlanSpec
            build_plan(spec): build the Plan (it fetches its twiddles)
            PlanCache.store, then execute
```

- **Problem:** `(size, target)`, the cache key. Direction isn't part of it: one plan runs both directions.
- **Config:** `CPUConfig` or `GPUConfig`, how to run it. `PlanSpec` is a size plus one of them.
- **Plan:** the ready-to-run object (as in FFTW), built from a `PlanSpec` for one size and implemented by `CPUPlan` or `GPUPlan`. Nothing is compiled at runtime: it picks among kernels compiled into the library. It shares its twiddle tables with that backend's table cache.

All of the planning is in `plan/planner.h`: `EstimateParams`, `CPUPlanner`, `GPUPlanner`, and the thin `Planner` that handles the cache and `Auto`. None of it needs CUDA, so the GPU rules are tested in every build.

## Estimate rules

Estimate only; there is no `Measure` mode yet. The inputs are detected hardware (`SystemConfig`, where unknown values are 0 and fall back to defaults) and `EstimateParams`.

| Decision | Rule |
|---|---|
| Direct DFT | `N <= dft_max` (8) |
| Four-step split | none while `N <= tile`, split above |
| CPU tile | largest power of 2 with `16·tile <= l2_fraction · L2` |
| Split size | `N2 = min(tile, 2^ceil(log2 N / 2))`, `N1 = N / N2` |
| Radix | 4 when N is a power of 4, else 2 |
| Threads | physical cores when OpenMP is on and `N·log2 N >= parallel_work_per_core · cores`, else 1 |
| SIMD | highest detected level |
| `Auto` | GPU when CUDA is usable and `N >= gpu_min` (2^16); below that, host/device copies cost more than the FFT |
| GPU | shared-memory Stockham, radix 2. Longest column L = largest power of 2 with `256 B·L <= opt-in smem per block`, `4L <= max threads per block` (one thread per butterfly, also the block size) and >= 2 resident blocks per SM by smem and threads. `passes = ceil(log2 N / log2 L)`, log2 N spread evenly over the passes |

**Four-step** is not a separate algorithm: `CPUConfig::split` (0 = off) wraps the chosen traversal. Transpose, N2 row FFTs of length N1, twiddle by `W_N^{n2·k1}`, transpose, N1 row FFTs of length N2, transpose. Row loops run in parallel and each row FFT is serial. Twiddles come from two √N-entry tables. There's one level of split, so rows stay in cache up to `N = tile²`.

**Why the split is balanced:** with N2 set to the full tile, N1 was 2 at 2^20 on the M3. The first pass then ran half a million length-2 FFTs, and four-step was 2.7× slower than iterative.

## Measurements

Apple M3 (8 cores, 16 MiB L2, NEON), `fftop_bench`, ms per FFT.

| N | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| direct DFT | 0.00002 | 0.00002 | 0.00005 | 0.00019 | 0.00079 | 0.00377 |
| iterative (best radix) | 0.00007 | 0.00007 | 0.00022 | 0.00017 | 0.00049 | 0.00039 |

| N | iterative, 1 thread | four-step, 1 thread | iterative, 8 threads | four-step, 8 threads |
|---|---|---|---|---|
| 2^16 | 0.61 | 0.95 | 0.57 | 0.40 |
| 2^18 | 2.72 | 6.91 | 2.20 | 2.14 |
| 2^20 | 17.5 | 21.9 | 9.76 | 5.56 |
| 2^21 | 63.3 | 64.9 | 51.7 | 17.3 |
| 2^22 | 94.9 | 134 | 83.5 | 35.2 |
| 2^23 | 350 | 268 | 323 | 72.0 |

- DFT wins through N = 8, hence `dft_max = 8`.
- The CPU tile on this machine is 2^19. With threads, four-step wins from 2^16 on. With one thread it loses until 2^23, probably because the transposes on power-of-2 strides cause cache-set conflicts.
- Re-check on other machines before changing defaults: `./bench/bench-sweep.sh --only iterative-vs-four-step --only dft-vs-fft --only planner-vs-best`.

## Behaviour

- Inverse is unnormalized. Sizes must be powers of 2 (asserted).
- Only tests and the bench build configs by hand; an invalid config throws `std::invalid_argument`.
- Threads are set per run (`num_threads`), never globally.
- GPU plans keep their device buffers and stage host copies through pinned memory (`GPUConfig::pinned_staging`). Whether that beats pageable copies hasn't been measured.
- Deferred: thread safety and `Measure` mode (see `TODO.md`), plan-cache size limits, GPU-to-CPU fallback on out-of-memory, batched and multi-dimensional transforms. -->
