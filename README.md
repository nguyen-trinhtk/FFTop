# Optimize FFT implementation

This project's primary goal is to discover & benchmark different hardware-efficient implementations of the Fast Fourier Transform algorithm, using a mix-and-match of techniques while considering the computing architecture beneath. Currently the focus is on CPU implementations.

Please refer to [this document](./doc/optimizations.md) to learn more about each optimization. 

---

### Benchmarking
The last updated benchmark table for `N=4096`is shown below

| Variant | Time (miliseconds) |
|-------|---------------------:|
| Naive DFT | 324.638 |
| Radix-2 FFT (Cooley-Tukey) | 8.423 |

Please refer to [this log](./bench/results/bench.md) for the full benchmark snapshot.

#### CPU benchmarking
Benchmarking is done on my personal laptop, which uses a Silicon M3 chip with 4 cores of 4.05 GHz and 4 cores of 2.75 GHz CPU and 100GB/s memory bandwidth. The following compiler & flags are used:
```
g++-15 -g -O3 -march=native -std=c++17 -Wall -Wextra -I.
```

(and yes I'm fully aware this is not among the strongest setup for benchmarking).

#### TODO: GPU benchmarking

---

### Testing

All implemented variants are tested against naive DFT, which are ensured to be correct. The following tests are conducted: 

- Test correctness for input sizes 2^k
- Edge cases: 
    + All-zeros input
    + Impulse at 0
    + Impulse at k
- Parseval's theorem: total energy of signal in time domain equals to that of frequency domain