#pragma once

#include <complex>
#include <cstddef>
#include <vector>

// shared type definitions
namespace FFTop {
// TODO: other data type
using Real = double;
using Complex = std::complex<Real>;
using Buffer = std::vector<Complex>;

enum class HardwareTarget { Auto, CPU, GPU };
enum class Direction { Forward, Inverse };

enum class Radix : std::size_t { Radix2 = 2, Radix4 = 4 };

struct FFTOptions {
    HardwareTarget hardware_target = HardwareTarget::Auto;
    Direction direction = Direction::Forward;
};
}  // namespace FFTop
