// Shared radix-2 / radix-4 bodies. Include an isa_*.h first, then this file
// inside an anonymous namespace.

constexpr Real kTwoPi = Real(6.283185307179586476925286766559L);

ISA::Pack twiddle(std::size_t index, std::size_t order, Direction dir) {
    const Real sign  = (dir == Direction::Forward) ? Real(-1) : Real(1);
    const Real angle = sign * kTwoPi * Real(index) / Real(order);
    return ISA::setc(std::cos(angle), std::sin(angle));
}

void radix2(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    Complex* x0 = data.data() + offset;
    Complex* x1 = x0 + stride;

    const std::size_t order = 2 * stride;

    for (std::size_t k = 0; k < stride; ++k) {
        const ISA::Pack a0 = ISA::load(x0[k]);
        const ISA::Pack a1 = ISA::cmul(twiddle(k, order, dir), ISA::load(x1[k]));

        ISA::store(x0[k], ISA::add(a0, a1));
        ISA::store(x1[k], ISA::sub(a0, a1));
    }
}

void radix4(Buffer& data, std::size_t offset, std::size_t stride, Direction dir) {
    Complex* x0 = data.data() + offset;
    Complex* x1 = x0 + stride;
    Complex* x2 = x1 + stride;
    Complex* x3 = x2 + stride;

    const std::size_t order   = 4 * stride;
    const bool        forward = dir == Direction::Forward;

    for (std::size_t k = 0; k < stride; ++k) {
        const ISA::Pack a0 = ISA::load(x0[k]);
        const ISA::Pack a1 = ISA::cmul(twiddle(k, order, dir), ISA::load(x1[k]));
        const ISA::Pack a2 = ISA::cmul(twiddle(2 * k, order, dir), ISA::load(x2[k]));
        const ISA::Pack a3 = ISA::cmul(twiddle(3 * k, order, dir), ISA::load(x3[k]));

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
}
