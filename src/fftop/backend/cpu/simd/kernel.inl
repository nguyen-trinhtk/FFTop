// Shared radix-2 / radix-4 bodies
// Include an isa_*.h first, then this file
// inside an anonymous namespace.
//
// W is the full-N forward table: W[k] = cis(-2π k / N).
// Stage twiddle W_order^{m k} = W[m k · (N / order)]. Inverse is conj.
// A SIMD pack starting at k must use m * (k + lane), which is load_w(W, k, m * step).
// load_w(W, m * k, step) is only right for a one-wide kernel.

// Load twiddle factor
ISA::Pack load_w(const Complex* W, std::size_t k, std::size_t step, Direction dir) {
    ISA::Pack w;
    if constexpr (ISA::width == 1) {
        w = ISA::load(W[k * step]);
    } else if (step == 1) {
        w = ISA::load(W[k]);
    } else {
        alignas(64) Real lanes[2 * ISA::width];
        for (std::size_t i = 0; i < ISA::width; ++i) {
            const Complex z = W[(k + i) * step];
            lanes[2 * i]     = z.real();
            lanes[2 * i + 1] = z.imag();
        }
        w = ISA::load_pack(lanes);
    }
    return dir == Direction::Forward ? w : ISA::conj(w);
}

Complex twiddle_at(const Complex* W, std::size_t k, std::size_t step, Direction dir) {
    const Complex w = W[k * step];
    return dir == Direction::Forward ? w : std::conj(w);
}

void radix2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
            const Complex* W, std::size_t n) {
    Complex* x0 = data.data() + offset;
    Complex* x1 = x0 + stride;

    const std::size_t order = 2 * stride;
    const std::size_t step  = n / order;

    std::size_t k = 0;
    for (; k + ISA::width <= stride; k += ISA::width) {
        const ISA::Pack a0 = ISA::load(x0[k]);
        const ISA::Pack a1 = ISA::cmul(load_w(W, k, step, dir), ISA::load(x1[k]));

        ISA::store(x0[k], ISA::add(a0, a1));
        ISA::store(x1[k], ISA::sub(a0, a1));
    }
    for (; k < stride; ++k) {
        const Complex a0 = x0[k];
        const Complex a1 = twiddle_at(W, k, step, dir) * x1[k];
        x0[k] = a0 + a1;
        x1[k] = a0 - a1;
    }
}

void radix4(Buffer& data, std::size_t offset, std::size_t stride, Direction dir,
            const Complex* W, std::size_t n) {
    Complex* x0 = data.data() + offset;
    Complex* x1 = x0 + stride;
    Complex* x2 = x1 + stride;
    Complex* x3 = x2 + stride;

    const std::size_t order   = 4 * stride;
    const std::size_t step    = n / order;
    const bool        forward = dir == Direction::Forward;

    std::size_t k = 0;
    for (; k + ISA::width <= stride; k += ISA::width) {
        const ISA::Pack a0 = ISA::load(x0[k]);
        const ISA::Pack a1 = ISA::cmul(load_w(W, k,     step, dir), ISA::load(x1[k]));
        const ISA::Pack a2 = ISA::cmul(load_w(W, k, 2 * step, dir), ISA::load(x2[k]));
        const ISA::Pack a3 = ISA::cmul(load_w(W, k, 3 * step, dir), ISA::load(x3[k]));

        const ISA::Pack even_sum  = ISA::add(a0, a2);
        const ISA::Pack even_diff = ISA::sub(a0, a2);
        const ISA::Pack odd_sum   = ISA::add(a1, a3);
        const ISA::Pack odd_diff  = ISA::sub(a1, a3);
        const ISA::Pack odd_turn  = forward ? ISA::mul_minus_j(odd_diff)
                                            : ISA::mul_j(odd_diff);

        ISA::store(x0[k], ISA::add(even_sum, odd_sum));
        ISA::store(x1[k], ISA::add(even_diff, odd_turn));
        ISA::store(x2[k], ISA::sub(even_sum, odd_sum));
        ISA::store(x3[k], ISA::sub(even_diff, odd_turn));
    }
    for (; k < stride; ++k) {
        const Complex a0 = x0[k];
        const Complex a1 = twiddle_at(W, k,     step, dir) * x1[k];
        const Complex a2 = twiddle_at(W, k, 2 * step, dir) * x2[k];
        const Complex a3 = twiddle_at(W, k, 3 * step, dir) * x3[k];

        const Complex even_sum  = a0 + a2;
        const Complex even_diff = a0 - a2;
        const Complex odd_sum   = a1 + a3;
        const Complex odd_diff  = a1 - a3;
        const Complex odd_turn  = forward ? Complex{odd_diff.imag(), -odd_diff.real()}
                                          : Complex{-odd_diff.imag(), odd_diff.real()};

        x0[k] = even_sum + odd_sum;
        x1[k] = even_diff + odd_turn;
        x2[k] = even_sum - odd_sum;
        x3[k] = even_diff - odd_turn;
    }
}
