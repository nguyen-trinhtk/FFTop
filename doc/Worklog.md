# Worklog

A development of optimizations.

## Algorithmic
#### Cooley-Tukey's Fast Fourier Transform
The original Discrete Fourier Transform (DFT) has time complexity of $O(n^2)$. By partitioning the algorithm recursively on even and odd sub-data, Cooley and Tukey's divide-and-conquer algorithm achieved a $O(n \log n)$ complexity.

#### Iterative rewriting of Cooley-Tukey
The recursive algorithm was nice, but lack the ability to parallelize efficiently due to the dependency nature of recursion. Henceby, the iterative version is established, to support foreseeing the ops. Here each butterfly in the same stage is independent, and successive stages are dependent. Help speed-up due to scaling workload to multiple threads.

#### Four-step Bailey for big N
Target big N that couldn't fit in cache. Basically decompose 1D FFT with N-points into N=n1 * n2, where n1 and n2 would fit into cache. 

## CPU
#### Data structure optimizations
Theres a lot of opportunity to discover optimizations before moving on to parallel scaling, for example considering between AoS and SoA layout or precomputing twiddle table. 

#### Radix-4 blocking
50% Less data read. Possibly proceed to radix-8 or radix-16, but the benefits get exponentially less.

#### SIMD vectorization
Compress same butterfly ops on more butterfly with fewer instructions in a single instruction, multiple data manner. 

#### OpenMP parallelism
Again, the iterative Cooley-Tukey has exploit the opportunity to scale computation across core, so why not do it? Note that synchronization is required after independent threads handle all butterflies in one stage.

## GPU
#### Porting over iterative Cooley-Tukey
...

#### Stockham's FFT for memory-coalescing
Basically store an intermediate buffer to make memory ... contiguous, which support better memory-coalescing. The reason is GPU architecture is naturally (...)

#### Shared-memory tiling
Shared mem for faster data accessing

#### TODO: Warp shuffling
#### TODO: Lower-precision
TODO: look at error vs throughput tradeoff. Try emulation techniques to utilize tensor core (inspired by ozaki for cugemm).
