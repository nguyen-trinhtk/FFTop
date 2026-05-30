# Progressive optimization walkthrough

### I. Naive DFT

This is the foundational implementation of Fourier Transform, with time complexity of $O(n^2)$. Without changing the time complexity yet, this can be optimized with better exploit of parallelism and usage of cache; however we would not focus on that due to a higher impact algorithmic update in FFT.

### II. Cooley-Tukey FFT (radix-2)
This recursive algorithm reduces asymptotic complexity from $O(n^2)$ to $O(n \log{n})$, which made room for substantial performance gains especially over large input. However there will be better implementation that utilizes the hardware more efficiently.

### III. Radix-4


### TODO: Remarks on FFTW planner for radix selection
- Director (planner) vs executor pattern
- Adapt to hardware (treat as parameter)

