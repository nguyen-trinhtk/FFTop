<!-- TODO: diagram

### Basics: 1D FFT
- Radix (R): a group of input that is processed per stage. For now the code only support radix-2 and radix-4.
- Level/stage (L): a "row" in the butterfly network (this is not the same with row in 2D fft). Level i will have stride S=R^i.
- Block (B): a tuple of inputs that are processed together in the same stage that are S indices apart. 
    - Butterfly: regard to a 2x2 switch (which is a block itself in a radix-2 network). It has the shape like (TODO). For example, each radix-4 block has four butterflies (TODO)
- Group (G): a set of S blocks in a stage. As a consequent, these groups are consecutive. 

### Stockham FFT

I don't think theres anything here

### 2D FFT
- TODO: Row, column, tile, panel????? -->
