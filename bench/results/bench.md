# Benchmarks
Timestamp: 26-06-26 06:42:56
### Radix-2 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 5637.589 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 5587.797 |

### Radix-4 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 3587.082 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 3588.488 |

### Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 985.898 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 989.117 |

### SIMD Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 846.803 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 863.770 |

### OpenMP+SIMD Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 819.264 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 819.782 |

### Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 952.152 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 943.000 |

### Parallel Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 234.724 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 248.329 |

### FFTW3 (cold)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 399.843 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 391.609 |

### FFTW3 (steady)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 261.732 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 254.585 |

