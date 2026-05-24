# Basic FFT theory

### I. Fourier Transform
- Goal: transform data in time domain into frequency domain
- Used in signal processing, etc

### II. Discrete Fourier Transform
- Formula: 
$$
X_k = \sum_{n = 0}^{N - 1} x_n \cdot e^{-\frac{ 2 \pi i }{N} kn}
$$

- Using Euler's formula: $e^{-i \theta} = \cos{\theta} - i \sin{\theta}$
$$
X_k = \sum_{n = 0}^{N - 1} x_n \cdot \cos{\frac{ 2 \pi }{N} kn} - i \sin{\frac{ 2 \pi }{N} kn}
$$
