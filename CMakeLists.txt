cmake_minimum_required(VERSION 3.20)

project(FFTop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fftop STATIC
    src/fftop/backend/cpu/cpu.cpp
    src/fftop/backend/cpu/radix.cpp
    src/fftop/backend/gpu/gpu_backend.cpp
    src/fftop/backend/gpu/naive.cpp
    src/fftop/system/config.cpp
    src/fftop/fft.cpp
    src/fftop/plan/cache.cpp
    src/fftop/plan/planner.cpp
)

add_library(fftop::fftop ALIAS fftop)

target_include_directories(fftop
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

option(FFTOP_BUILD_TESTS "Build unit tests" ON)
option(FFTOP_BUILD_BENCH "Build benchmarks" ON)
option(FFTOP_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

if(FFTOP_ENABLE_CUDA)
    target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_CUDA)
endif()

if(FFTOP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FFTOP_BUILD_BENCH)
    add_subdirectory(bench)
endif()
