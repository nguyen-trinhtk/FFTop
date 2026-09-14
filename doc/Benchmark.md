# Benchmark

Each figure is one question: two variants, one axis, everything else held fixed.
C++ only times a spec. `bench/config/default.yaml` names the questions and axis defaults.

```bash
./bench/bench-sweep.sh
./bench/bench-sweep.sh --only radix2-vs-radix4
./bench/bench-sweep.sh --dry-run
```

Writes `log/<datetime>/system.json`, `bench.csv`, a copy of the config, and
one PNG per comparison. Replot without re-timing:

```bash
python bench/bench.py --plot log/<datetime>/bench.csv
```

`defaults.xscale` / `defaults.yscale` are `log` or `linear` (`scalar` = linear).
A comparison can override them. So can the CLI:

```bash
python bench/bench.py --plot log/<stamp>/bench.csv --yscale log
./bench/bench-sweep.sh --only naive-vs-fft --xscale log --yscale scalar
```

Each point is one warmup, one probe, then enough timed repeats to target
~100 ms of work (clamped to 20–5000). The `repeats` column records how many
were used. Shared variants are measured once and sliced into figures.

## CPU comparisons

| id | question | hold fixed | vary |
|---|---|---|---|
| `naive-vs-fft` | Is the fast algorithm actually faster? | — | naive vs recursive r2 scalar, \(N\le 2^{12}\) |
| `recursive-vs-iterative` | Is the loop version faster than recursion? | r2, scalar, serial | traversal |
| `radix2-vs-radix4` | Is grouping four points cheaper than two? | iterative, scalar, serial | radix, \(N=4^p\) only |
| `scalar-vs-simd` | Does vectorizing the kernel help? | iterative, r2, serial | ISA (this host) |
| `serial-vs-openmp` | Do extra threads help when N is large? | iterative, r2, scalar | threads, large N |
| `conclusion` | How much did the CPU work buy us? | — | iterative r2 scalar vs best |

SIMD on one machine is scalar vs the compiled host kernel (NEON here, AVX2 on
x86). Overlay two `bench.csv` files to put NEON and AVX2 on the same figure.

OpenMP writes `serial-vs-openmp.png` (time vs N) and
`serial-vs-openmp_speedup.png`.

`best` is iterative, radix-4 when \(N=4^p\) else radix-2, host SIMD, and
OpenMP at `cpu_threads` when the build has it.

## TODO: GPU kernels
