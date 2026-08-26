cmake_minimum_required(VERSION 3.20)

project(FFTop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fftop STATIC
    src/fftop/backend/cpu/cpu.cpp
    src/fftop/backend/cpu/butterfly.cpp
    src/fftop/backend/cpu/simd/kernel_scalar.cpp
    src/fftop/backend/cpu/simd/kernel_avx2.cpp
    src/fftop/backend/cpu/simd/kernel_neon.cpp
    src/fftop/backend/cpu/traversal_strategy.cpp
    src/fftop/backend/gpu/gpu_backend.cpp
    src/fftop/backend/gpu/naive.cpp
    src/fftop/backend/gpu/cooley_tukey.cpp
    src/fftop/system/config.cpp
    src/fftop/system/cpu.cpp
    src/fftop/system/gpu.cpp
    src/fftop/system/openmp.cpp
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
option(FFTOP_ENABLE_CUDA "Enable CUDA GPU backend if a CUDA compiler is found" ON)
option(FFTOP_ENABLE_OPENMP "Enable OpenMP parallelism" ON)

if(FFTOP_ENABLE_OPENMP)
    # AppleClang has no bundled OpenMP; use Homebrew libomp when present.
    if(APPLE AND NOT OpenMP_CXX_FOUND)
        foreach(_fftop_omp_prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
            if(EXISTS "${_fftop_omp_prefix}/lib/libomp.dylib")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${_fftop_omp_prefix}/lib/libomp.dylib")
                include_directories("${_fftop_omp_prefix}/include")
                break()
            endif()
        endforeach()
        unset(_fftop_omp_prefix)
    endif()
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fftop PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_OPENMP)
    else()
        message(WARNING "OpenMP not found; building without FFTOP_ENABLE_OPENMP")
        set(FFTOP_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP parallelism" FORCE)
    endif()
endif()

if(MSVC)
    set_source_files_properties(src/fftop/backend/cpu/simd/kernel_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
endif()

if(FFTOP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75)
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        target_sources(fftop PRIVATE
            src/fftop/backend/gpu/naive.cu
            src/fftop/backend/gpu/gpu_fft.cu
        )
        target_link_libraries(fftop PUBLIC CUDA::cudart)
        target_compile_definitions(fftop PRIVATE FFTOP_ENABLE_CUDA)
        set_target_properties(fftop PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
        )
        message(STATUS "FFTop: CUDA GPU backend enabled")
    else()
        message(STATUS "FFTop: CUDA compiler not found; building CPU-only")
        set(FFTOP_ENABLE_CUDA OFF CACHE BOOL
            "Enable CUDA GPU backend if a CUDA compiler is found" FORCE)
    endif()
endif()

if(FFTOP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(FFTOP_BUILD_BENCH)
    add_subdirectory(bench)
endif()
