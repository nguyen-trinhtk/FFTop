#include "fft/cpu/detail/simd_radix2.h"

#include "fft/cpu/simd-butterfly.h"
#include "fft/core/bitops.h"

namespace FFTCpu {
namespace detail {

void simd_radix2_inplace(std::vector<FFTCore::Complex>& data, const bool parallel_groups) {
    if (data.size() <= 1) {
        return;
    }
    FFTCore::bit_reverse_permute(data);
    FFTSimd::detail::fft_stages(data, parallel_groups);
}

}  // namespace detail
}  // namespace FFTCpu
