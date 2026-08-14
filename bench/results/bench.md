# Benchmarks
Timestamp: 26-08-14 12:34:11
### Radix-2 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.272 |
| Input sizes | 4096 | 3 | 1.226 |
| Input sizes | 16384 | 3 | 5.399 |
| Input sizes | 65536 | 3 | 23.493 |
| Input sizes | 262144 | 3 | 100.829 |
| Input sizes | 1048576 | 3 | 446.340 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 8151.123 |

### Radix-4 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.158 |
| Input sizes | 4096 | 3 | 1.446 |
| Input sizes | 16384 | 3 | 3.099 |
| Input sizes | 65536 | 3 | 14.121 |
| Input sizes | 262144 | 3 | 65.363 |
| Input sizes | 1048576 | 3 | 283.089 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 4882.285 |

### Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.019 |
| Input sizes | 4096 | 3 | 0.391 |
| Input sizes | 16384 | 3 | 0.443 |
| Input sizes | 65536 | 3 | 2.091 |
| Input sizes | 262144 | 3 | 10.202 |
| Input sizes | 1048576 | 3 | 52.901 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 1365.656 |

### OpenMP Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.391 |
| Input sizes | 4096 | 3 | 0.653 |
| Input sizes | 16384 | 3 | 1.006 |
| Input sizes | 65536 | 3 | 3.068 |
| Input sizes | 262144 | 3 | 9.213 |
| Input sizes | 1048576 | 3 | 44.538 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 1208.841 |

### Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.019 |
| Input sizes | 4096 | 3 | 0.154 |
| Input sizes | 16384 | 3 | 0.959 |
| Input sizes | 65536 | 3 | 3.370 |
| Input sizes | 262144 | 3 | 14.704 |
| Input sizes | 1048576 | 3 | 64.697 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 1175.726 |

### Parallel Four-Step FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.399 |
| Input sizes | 4096 | 3 | 0.343 |
| Input sizes | 16384 | 3 | 0.457 |
| Input sizes | 65536 | 3 | 1.305 |
| Input sizes | 262144 | 3 | 5.535 |
| Input sizes | 1048576 | 3 | 24.470 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 424.798 |

### FFTW3 (cold)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.015 |
| Input sizes | 4096 | 3 | 0.197 |
| Input sizes | 16384 | 3 | 0.356 |
| Input sizes | 65536 | 3 | 1.447 |
| Input sizes | 262144 | 3 | 6.725 |
| Input sizes | 1048576 | 3 | 28.261 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 568.363 |

### FFTW3 (steady)
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 1024 | 3 | 0.004 |
| Input sizes | 4096 | 3 | 0.019 |
| Input sizes | 16384 | 3 | 0.105 |
| Input sizes | 65536 | 3 | 0.547 |
| Input sizes | 262144 | 3 | 2.449 |
| Input sizes | 1048576 | 3 | 15.776 |
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Steady state | 16777216 | 3 | 405.070 |

