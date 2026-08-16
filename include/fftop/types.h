#pragma once

#include <complex>
#include <vector>

namespace FFTop {

using Real    = double;
using Complex = std::complex<Real>;
using Buffer  = std::vector<Complex>;

enum class Backend   { Auto, CPU, GPU };
enum class Direction { Forward, Inverse };

struct FFTOptions {
    Backend backend   = Backend::Auto;
    Direction direction = Direction::Forward;
};

}  // namespace FFTop
