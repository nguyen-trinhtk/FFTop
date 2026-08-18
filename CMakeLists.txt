cmake_minimum_required(VERSION 3.20)

project(FFTop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fftop STATIC
    src/fftop/backend/cpu/cpu.cpp
    src/fftop/backend/cpu/butterfly.cpp
    src/fftop/backend/cpu/simd/radix2_scalar.cpp
    src/fftop/backend/cpu/traversal_strategy.cpp
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
option(FFTOP_ENABLE_OPENMP "Enable OpenMP parallelism" ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    option(FFTOP_ENABLE_NEON "Enable NEON SIMD kernels" ON)
else()
    option(FFTOP_ENABLE_NEON "Enable NEON SIMD kernels" OFF)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i.86")
    option(FFTOP_ENABLE_AVX2 "Enable AVX2 SIMD kernels" ON)
else()
    option(FFTOP_ENABLE_AVX2 "Enable AVX2 SIMD kernels" OFF)
endif()

if(FFTOP_ENABLE_OPENMP)
    # AppleClang ships without OpenMP; point CMake at Homebrew's libomp.
    if(APPLE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
        include_directories(/opt/homebrew/opt/libomp/include)
    endif()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(fftop PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_OPENMP)
endif()

if(FFTOP_ENABLE_NEON)
    target_sources(fftop PRIVATE src/fftop/backend/cpu/simd/radix2_neon.cpp)
    target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_NEON)
endif()

if(FFTOP_ENABLE_AVX2)
    target_sources(fftop PRIVATE src/fftop/backend/cpu/simd/radix2_avx2.cpp)
    target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_AVX2)
    if(MSVC)
        set_source_files_properties(src/fftop/backend/cpu/simd/radix2_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/fftop/backend/cpu/simd/radix2_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

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
