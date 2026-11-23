<!-- # BLOCK 3 HOURS SATURDAY CODE REVIEW
- all: 
  + radix-B
  + generalized N
  + four-step 2D FFT
  + NUMA aware?
  + Twiddle table

  + benchmark to cache plan???
  + maybe let user do a warmup discovery before needing to run any fft
  - In-place support: FFT(vec) instead of out = FFT(in)
  - Technically if we do radix-2 and radix-3 then we can do everything.
  - Cache network construction butterflies???
  - Thread safety of cached plans: two threads may run the same plan at once.
    CPU: per-call scratch, already safe. GPU: reuses device/pinned buffers, so calls
    are serialized by a lock; per-thread buffers or per-call streams would remove it.
  - Planner: Measure mode + saved results keyed by a hardware fingerprint (doc/planner-design.md)
  - Four-step transposes on power-of-2 strides likely hit cache-set conflicts (pad or buffer)

- cpu: 
  + add sse2 for full compatibility, avx, (avx512 later)

- gpu:
  + stockham
  + tiling, bailey 4-step

  + coalesced mem access (+ transpose for 2D FFTs or 4 step FFTs)
  + register-level
  + warp shuffle (mem in same warp directly xchange without going thru shared mem)
  + avoid explicit bit reversal (???)
  
  + lower-precision modes
  + stage fusion -->