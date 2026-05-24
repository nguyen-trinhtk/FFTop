# Basic FFT theory

## Preliminaries

### 1. Convolution

Given two polynomials
$$  
    f(x) = a_0 + a_1x + \cdots + a_n x^n \\
    g(x) = b_0 + b_1x + \cdots + b_m x^m
$$

represented as vector of length $N \ge n + m + 1$
$$  
    f = (a_0, a_1, \cdots, a_n, 0, 0, \cdots, 0) \\
    g = (b_0, b_1, \cdots, b_n, 0, 0, \cdots, 0)
$$
The product of $f(x)$ and $g(x)$ would be equal to the convolution of vectors $f$ and $g$, defined as below: 
$$
    f(x) \times g(x) = f \otimes g = (a_0b_0, a_1b_0 + a_0b_1, a_2b_0 + a_1b_1 + _0b_2 \cdots)
$$

### 2. 

## Fourier Transform
- Goal: transform data in time domain into frequency domain
- Used in signal processing, etc

### 1. Discrete Fourier Transform
- Formula: 
$$
X_k = \sum_{n = 0}^{N - 1} x_n \cdot e^{-\frac{ 2 \pi i }{N} kn}
$$

- Using Euler's formula: $e^{-i \theta} = \cos{\theta} - i \sin{\theta}$
$$
X_k = \sum_{n = 0}^{N - 1} x_n \cdot \cos{\frac{ 2 \pi }{N} kn} - i \sin{\frac{ 2 \pi }{N} kn}
$$


### References
- Cornell University's CS6820 lecture notes: https://www.cs.cornell.edu/courses/cs6820/2022fa/Handouts/FFT.pdf