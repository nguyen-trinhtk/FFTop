include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(fftop_tests
    test_fft.cpp
    test_backend.cpp
    test_planner.cpp
    test_system.cpp
)
target_link_libraries(fftop_tests PRIVATE fftop::fftop GTest::gtest_main)
target_include_directories(fftop_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(fftop_tests)
