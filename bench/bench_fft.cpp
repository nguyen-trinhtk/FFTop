#include "fftop/backend/backend_registry.h"
#include "fftop/plan/plan.h"
#include "fftop/system.h"
#include "fftop/types.h"

#include <chrono>
#include <iostream>

namespace {

void bench(FFTop::IBackend& backend, const FFTop::FFTPlan& plan) {
    constexpr int repeats = 20;
    FFTop::Buffer input(plan.size, {1, 0});
    FFTop::Buffer output;
    backend.execute(plan, input, output);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) backend.execute(plan, input, output);
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    const int radix = plan.radix == FFTop::RadixPolicy::Radix4 ? 4 : 2;
    std::cout << backend.name() << ',' << radix << ',' << plan.size << ',' << ms << '\n';
}

}  // namespace

int main() {
    std::cerr << FFTop::describe_system(FFTop::system_config());
    std::cout << "backend,radix,size,ms\n";
    // Powers of four, the sizes both radices can tile, so the two are comparable.
    for (std::size_t n = 64; n <= 4096; n *= 4) {
        for (FFTop::RadixPolicy radix : {FFTop::RadixPolicy::Radix2, FFTop::RadixPolicy::Radix4}) {
            FFTop::FFTPlan plan;
            plan.size    = n;
            plan.backend = FFTop::Backend::CPU;
            plan.radix   = radix;
            auto cpu = FFTop::make_cpu_backend(plan);
            bench(*cpu, plan);
        }
    }
}
