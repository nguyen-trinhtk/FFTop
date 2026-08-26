- all: 
  + radix-B
  + generalized N
  + four-step 2D FFT
  + NUMA aware?
  + Twiddle table
- cpu: 
  + plan
  + add sse2 for full compatibility, avx, (avx512 later)
- gpu: 
  + naive-only for now
  + coalesced mem access (+ transpose for 2D FFTs or 4 step FFTs)
  + shared mem tiling
  + register-level
  + warp shuffle (mem in same warp directly xchange without going thru shared mem)
  + avoid explicit bit reversal (???)
  + stage fusion


# TODO
- Generalized N
- Four-step N1 x N2 planning
- In-place support: FFT(vec) instead of out = FFT(in)
- Technically if we do radix-2 and radix-3 then we can do everything.
- NUMA aware
- Cache network construction butterflies???