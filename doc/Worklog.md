# Worklog

A running development of optimizations.

## Algorithmic
#### Cooley-Tukey's Fast Fourier Transform
The original Discrete Fourier Transform (DFT) has time complexity of $O(n^2)$. By partitioning the algorithm recursively on even and odd sub-data, Cooley and Tukey's divide-and-conquer algorithm achieved a $O(n \log n)$ complexity.

#### Iterative rewriting of Cooley-Tukey
The recursive algorithm was nice, but lack the ability to parallelize efficiently due to the dependency nature of recursion. Henceby, the iterative version is established, to support foreseeing the ops more clearly. Here each butterfly in the same stage is independent, and successive stages are dependent. Help speed-up due to scaling workload out to multiple threads.

#### Four-step Bailey for big N
This variant targets big N that couldn't fit in cache otherwise. Basically, the idea is to decompose 1D FFT with $N$ points into a 2D FFT of size $n_1 \times n_2$, where $n_1$ and $n_2$ would fit into cache. 

## CPU
#### Data structure optimizations
There's a lot of opportunity to discover optimizations here before moving on to parallel scaling, for example considering between AoS and SoA layout or precomputing the twiddle table. 

#### Radix-4 blocking
Theoretically, by doing the twiddling on 4 elements at a time instead of 2, we'd be able to save 50% less data read. It is possible that we do a higher radix, but as the benefits gets exponentially less and memory complication gradually dominates, we'd reach a diminishing return at some point.

#### SIMD vectorization
Basically we are dealing with the same twiddle operations across butterflies, just with different data, hence this is the perfect chance to exploit hardware's SIMD capability. 

#### OpenMP parallelism
Again, the iterative Cooley-Tukey has already exploit the opportunity to scale computation across core, so why not do it? Note that synchronization is required after independent threads handle all butterflies in one stage.

## GPU
For GPU I mainly deals with NVIDIA GPU architectures.
#### Porting over iterative Cooley-Tukey
Reimplementing Cooley-Tukey in GPU. Note that GPU thread loads data in a SIMT warps (and yes memory-coalescing is important), so we can achieve better performance via the Stockham variant discussed below.

#### Stockham's FFT for memory-coalescing
This is sometimes referred to as ping-pong buffering. Basically, store an intermediate buffer to make memory accesses more contiguous, then juggle the indexing back and forth between buffers. The tradeoff is extra memory for no bit-reversal permutation, which supports better memory coalescing on the GPU.

#### Shared-memory tiling
Split N points into tile blocks and load each tile into per-block shared memory for faster on-chip data accessing.

#### TODO: Warp shuffling
#### TODO: Lower-precision
TODO: look at the error vs throughput tradeoff. Try emulation techniques to utilize tensor core (inspired by ozaki for cugemm).
