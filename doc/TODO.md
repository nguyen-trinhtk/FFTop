# BIG CODE REVIEW THIS SATURDAY, BLOCK 3 HOURS PLEASE
- all: 
  + radix-B
  + generalized N
  + four-step 2D FFT
  + NUMA aware?
  + Twiddle table

  + benchmark to cache plan???
  + maybe let user do a warmup discovery before needing to run any fft

- cpu: 
  + add sse2 for full compatibility, avx, (avx512 later)

- gpu: 
  + naive-only for now
  + then tiling, stockham, bailey 4-step
  + shared mem tiling

  + coalesced mem access (+ transpose for 2D FFTs or 4 step FFTs)
  + register-level
  + warp shuffle (mem in same warp directly xchange without going thru shared mem)
  + avoid explicit bit reversal (???)
  
  + lower-precision modes
  + stage fusion


# TODO
- Generalized N
- Four-step N1 x N2 planning
- In-place support: FFT(vec) instead of out = FFT(in)
- Technically if we do radix-2 and radix-3 then we can do everything.
- NUMA aware
- Cache network construction butterflies???