#pragma once

#include "fftop/main/types.h"

namespace FFTop {
Buffer fft(const Buffer& input, const FFTOptions& options = {});
}  // namespace FFTop
