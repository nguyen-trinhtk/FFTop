#pragma once
// Radix-2 and radix-4 butterflies
// Calls operations provided by each SIMD kernel

#include "fftop/backend/cpu/simd/kernels.h"

namespace FFTop::CPU
{
    namespace
    {
        // Load a pack of twiddles
        template <class Kernel>
        typename Kernel::Pack load_w(const Complex *Ws, std::size_t k, Direction dir)
        {
            const typename Kernel::Pack w = Kernel::load(Ws[k]);
            return dir == Direction::Forward ? w : Kernel::conj(w); // conjugate for inverse FFT
        }

        // Helper for scalar tails
        // Single twiddle load
        template <class Kernel>
        Complex twiddle_at(const Complex *Ws, std::size_t k, Direction dir)
        {
            const Complex w = Ws[k];
            return dir == Direction::Forward ? w : Complex{w.real(), -w.imag()};
        }
        // Single element multiply
        template <class Kernel>
        Complex mul(Complex a, Complex b)
        {
            return {a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real()};
        }
    } // namespace

    template <class Kernel>
    void radix2(Buffer &data, std::size_t offset, std::size_t stride, Direction dir, const Complex *Ws)
    {
        using Pack = typename Kernel::Pack;

        Complex *x0 = data.data() + offset;
        Complex *x1 = x0 + stride;

        std::size_t k = 0;
        for (; k + Kernel::width <= stride; k += Kernel::width)
        {
            // do as many simd as possible
            const Pack a0 = Kernel::load(x0[k]);
            const Pack a1 = Kernel::cmul(load_w<Kernel>(Ws, k, dir), Kernel::load(x1[k]));

            Kernel::store(x0[k], Kernel::add(a0, a1));
            Kernel::store(x1[k], Kernel::sub(a0, a1));
        }
        for (; k < stride; ++k)
        {
            // the rest: scalar
            const Complex a0 = x0[k];
            const Complex a1 = mul<Kernel>(twiddle_at<Kernel>(Ws, k, dir), x1[k]);
            x0[k] = a0 + a1;
            x1[k] = a0 - a1;
        }
    }

    template <class Kernel>
    void radix4(Buffer &data, std::size_t offset, std::size_t stride, Direction dir,
                const Complex *Ws) {
        using Pack = typename Kernel::Pack;

        Complex *x0 = data.data() + offset;
        Complex *x1 = x0 + stride;
        Complex *x2 = x1 + stride;
        Complex *x3 = x2 + stride;

        const Complex *W1 = Ws; // W^k
        const Complex *W2 = Ws + stride; // W^2k
        const Complex *W3 = Ws + 2 * stride; // W^3k
        const bool forward = (dir == Direction::Forward);

        std::size_t k = 0;
        for (; k + Kernel::width <= stride; k += Kernel::width)
        {
            // as many vectorized processing as possible
            const Pack a0 = Kernel::load(x0[k]);
            const Pack a1 = Kernel::cmul(load_w<Kernel>(W1, k, dir), Kernel::load(x1[k])); // w * x1
            const Pack a2 = Kernel::cmul(load_w<Kernel>(W2, k, dir), Kernel::load(x2[k])); // w^2 * x2 
            const Pack a3 = Kernel::cmul(load_w<Kernel>(W3, k, dir), Kernel::load(x3[k])); // w^3 * x3

            const Pack even_sum = Kernel::add(a0, a2);
            const Pack even_diff = Kernel::sub(a0, a2);
            const Pack odd_sum = Kernel::add(a1, a3);
            const Pack odd_diff = Kernel::sub(a1, a3);
            
            const Pack odd_turn = forward ? Kernel::mul_minus_j(odd_diff)
                                          : Kernel::mul_j(odd_diff);

            Kernel::store(x0[k], Kernel::add(even_sum, odd_sum));
            Kernel::store(x1[k], Kernel::add(even_diff, odd_turn));
            Kernel::store(x2[k], Kernel::sub(even_sum, odd_sum));
            Kernel::store(x3[k], Kernel::sub(even_diff, odd_turn));
        }
        for (; k < stride; ++k)
        {
            // scalar tail
            const Complex a0 = x0[k];
            const Complex a1 = mul<Kernel>(twiddle_at<Kernel>(W1, k, dir), x1[k]);
            const Complex a2 = mul<Kernel>(twiddle_at<Kernel>(W2, k, dir), x2[k]);
            const Complex a3 = mul<Kernel>(twiddle_at<Kernel>(W3, k, dir), x3[k]);

            const Complex even_sum = a0 + a2;
            const Complex even_diff = a0 - a2;
            const Complex odd_sum = a1 + a3;
            const Complex odd_diff = a1 - a3;
            const Complex odd_turn = forward ? Complex{odd_diff.imag(), -odd_diff.real()}
                                             : Complex{-odd_diff.imag() odd_diff.real()};

            x0[k] = even_sum + odd_sum;
            x1[k] = even_diff + odd_turn;
            x2[k] = even_sum - odd_sum;
            x3[k] = even_diff - odd_turn;
        }
    }

} // namespace FFTop::CPU
