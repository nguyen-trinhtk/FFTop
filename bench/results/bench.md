# Benchmarks
Timestamp: 26-05-30 15:25:29
### Reference DFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 64 | 10 | 0.063 |
| Input sizes | 256 | 10 | 0.887 |
| Input sizes | 4096 | 10 | 183.721 |

### Radix-2 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 64 | 10 | 0.013 |
| Input sizes | 256 | 10 | 0.049 |
| Input sizes | 4096 | 10 | 0.887 |

### Radix-4 FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 64 | 10 | 0.007 |
| Input sizes | 256 | 10 | 0.028 |
| Input sizes | 4096 | 10 | 0.527 |

### Iterative In-Place FFT
| Benchmark | N | Runs | Avg ms |
|---|---:|---:|---:|
| Input sizes | 64 | 10 | 0.001 |
| Input sizes | 256 | 10 | 0.003 |
| Input sizes | 4096 | 10 | 0.071 |

