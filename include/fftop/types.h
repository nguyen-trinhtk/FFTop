#pragma once

#include <complex>
#include <vector>

// shared type definitions
namespace FFTop {

using Real    = double;
using Complex = std::complex<Real>;
using Buffer  = std::vector<Complex>;

enum class HardwareTarget { Auto, CPU, GPU };
enum class Direction      { Forward, Inverse };

struct FFTOptions {
    HardwareTarget hardware_target = HardwareTarget::Auto;
    Direction      direction       = Direction::Forward;
};
}  // namespace FFTop
