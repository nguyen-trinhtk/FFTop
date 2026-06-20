# Benchmarks
Timestamp: 26-06-14 12:42:01
### Radix-2 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 6658.602 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 5627.988 |

### Radix-4 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 3666.496 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 3662.237 |

### Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 1038.184 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 1038.356 |

### SIMD Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 943.841 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 938.455 |

### OpenMP+SIMD Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 881.973 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 881.267 |

### Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 1005.527 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 1013.769 |

### Parallel Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 297.540 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 300.761 |

### FFTW3 (cold)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 417.870 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 401.297 |

### FFTW3 (steady)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 16777216 | 3 | 493.894 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 387.356 |

