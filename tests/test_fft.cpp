#include "fftop/fft.h"
#include "support/check.h"

#include <gtest/gtest.h>

using namespace FFTop;
using namespace FFTop::Test;

TEST(Fft, MatchesDft) {
    auto transform = [](const Buffer& x, Direction dir) { return fft(x, {Backend::CPU, dir}); };
    for_each_input(kPowersOfTwo, [&](const Buffer& x) { expect_matches_dft(transform, x); });
}
