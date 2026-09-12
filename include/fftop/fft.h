#pragma once

#include "fftop/types.h"

namespace FFTop {
// Public interface
Buffer fft(const Buffer& input, const FFTOptions& options = {});
}  // namespace FFTop
