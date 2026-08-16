#pragma once

#include "fftop/types.h"

namespace FFTop {

Buffer fft(const Buffer& input, const FFTOptions& options = {});

}  // namespace FFTop
