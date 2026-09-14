add_executable(fftop_bench bench_fft.cpp)
target_link_libraries(fftop_bench PRIVATE fftop::fftop)
target_include_directories(fftop_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/tests
)

# Thread counts in the spec must match the runtime (omp_set_num_threads).
if(OpenMP_CXX_FOUND)
    target_link_libraries(fftop_bench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(fftop_bench PRIVATE FFTOP_ENABLE_OPENMP)
endif()
