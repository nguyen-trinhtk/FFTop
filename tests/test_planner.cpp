#include "fftop/plan/planner.h"

#include <gtest/gtest.h>

using namespace FFTop;

TEST(Planner, PrefersRadix4WhenSizeIsPowerOfFour) {
    Planner planner({});
    EXPECT_EQ(planner.make_plan(256, {}).radix, RadixPolicy::Radix4);
    EXPECT_EQ(planner.make_plan(32, {}).radix, RadixPolicy::Radix2);
}

TEST(Planner, AutoPicksGpuWhenCudaIsUsable) {
    SystemConfig sys;
    sys.cuda = true;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).hardware_target, HardwareTarget::GPU);
    EXPECT_EQ(Planner(sys).make_plan(16, {HardwareTarget::CPU, Direction::Forward})
                  .hardware_target,
              HardwareTarget::CPU);
}

TEST(Planner, AutoStaysOnCpuWithoutCuda) {
    SystemConfig sys;
    sys.nvidia_gpu = true;
    sys.cuda       = false;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).hardware_target, HardwareTarget::CPU);
}
