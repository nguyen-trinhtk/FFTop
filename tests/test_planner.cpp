#include "fftop/plan/planner.h"

#include <gtest/gtest.h>

using namespace FFTop;

TEST(Planner, PrefersRadix4WhenSizeIsPowerOfFour) {
    Planner planner({});
    EXPECT_EQ(planner.make_plan(256, {}).radix, RadixPolicy::Radix4);
    EXPECT_EQ(planner.make_plan(32, {}).radix, RadixPolicy::Radix2);
}

TEST(Planner, ParallelOnlyWithOpenMp) {
    SystemConfig sys;
    sys.cpu_threads = 8;
    sys.openmp = false;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).execution, Execution::Serial);
    sys.openmp = true;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).execution, Execution::Parallel);
}

TEST(Planner, AutoPicksGpuWhenCudaIsUsable) {
    SystemConfig sys;
    sys.cuda = true;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).backend, Backend::GPU);
    EXPECT_EQ(Planner(sys).make_plan(16, {Backend::CPU, Direction::Forward}).backend,
              Backend::CPU);
}

TEST(Planner, AutoStaysOnCpuWithoutCuda) {
    SystemConfig sys;
    sys.nvidia_gpu = true;
    sys.cuda = false;
    EXPECT_EQ(Planner(sys).make_plan(16, {}).backend, Backend::CPU);
}
