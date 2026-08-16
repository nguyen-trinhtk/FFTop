#include "fftop/backend/gpu/naive.h"

namespace FFTop {

void NaiveGPUBackend::execute(const FFTPlan&, const Buffer&, Buffer&) {
    // TODO: naive O(N^2) DFT on GPU
}

}  // namespace FFTop
