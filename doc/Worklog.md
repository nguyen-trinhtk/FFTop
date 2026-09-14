# Worklog

A development of optimizations.

## Algorithmic
#### Cooley-Tukey's Fast Fourier Transform
The original Discrete Fourier Transform (DFT) has time complexity of $O(n^2)$. By partitioning the algorithm recursively on even and odd sub-data, Cooley and Tukey's divide-and-conquer algorithm achieved a $O(n \log n)$ complexity.
#### Iterative rewriting of Cooley-Tukey
#### Four-step Bailey for big N
Same decomposition later: CPU uses cache-sized N1×N2; GPU uses an explicit transpose plus tiled inner FFTs.

## CPU
#### Data structure optimizations
Twiddle table, memory layout, etc.
#### Radix-4 blocking
#### SIMD vectorization
#### OpenMP parallelism
#### Four-step Bailey
Cache blocking for large N.

## GPU
#### Porting over iterative Cooley-Tukey
#### Stockham's FFT for memory-coalescing
#### Shared-memory tiling
#### Warp shuffling
#### Four-step Bailey
For N that does not fit in shared memory.
#### TODO: Lower-precision & tensor core emulation
