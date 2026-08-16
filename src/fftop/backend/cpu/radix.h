#pragma once

#include "fftop/types.h"

namespace FFTop::CPU {
void radix2(const Buffer& input, Buffer& output);
// TODO: radix-3, radix-4, etc. also inplace
}  // namespace FFTop::CPU
