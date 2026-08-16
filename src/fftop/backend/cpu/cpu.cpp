#include "fftop/backend/cpu/cpu.h"

namespace FFTop {

void CPUBackend::execute(const FFTPlan&, const Buffer&, Buffer&) {
    // TODO: dispatch to radix kernel based on plan
}

}  // namespace FFTop
