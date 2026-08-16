add_executable(fftop_bench bench_fft.cpp)
target_link_libraries(fftop_bench PRIVATE fftop::fftop)
target_include_directories(fftop_bench PRIVATE ${PROJECT_SOURCE_DIR}/src)
