# Override on the command line: make CXX=g++-15
CXX ?= g++
ifneq ($(wildcard /opt/homebrew/bin/g++-15),)
CXX := /opt/homebrew/bin/g++-15
endif

INCLUDES := -Iinclude
CXXFLAGS := -g -O3 -march=native -std=c++17 -Wall -Wextra $(INCLUDES)
FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null)
FFTW_LIBS := $(shell pkg-config --libs fftw3 2>/dev/null)
ifeq ($(FFTW_LIBS),)
FFTW_CFLAGS := -I/opt/homebrew/include
FFTW_LIBS := -L/opt/homebrew/lib -lfftw3
endif
OPENMP_FLAGS := -fopenmp
BENCH_CXXFLAGS := $(CXXFLAGS) $(FFTW_CFLAGS) $(OPENMP_FLAGS)
TEST_CXXFLAGS := $(CXXFLAGS) $(OPENMP_FLAGS)

TEST_BIN := build/run_all_tests
BENCH_BIN := build/run_all_benchmarks

FFT_SOURCES := \
	src/fft/ref/dft.cpp \
	src/fft/cpu/detail/iterative_radix2.cpp \
	src/fft/cpu/detail/matrix_ops.cpp \
	src/fft/cpu/detail/simd_radix2.cpp \
	src/fft/cpu/detail/four_step.cpp \
	src/fft/cpu/radix-2.cpp \
	src/fft/cpu/radix-4.cpp \
	src/fft/cpu/iterative.cpp \
	src/fft/cpu/simd-iter.cpp \
	src/fft/cpu/openmp-iter.cpp \
	src/fft/cpu/four-step.cpp \
	src/fft/cpu/parallel-four-step.cpp

TEST_SOURCES := \
	test/run_all_tests.cpp \
	test/utils.cpp \
	test/implementations.cpp \
	test/test_edge_cases.cpp \
	test/test_power_of_2.cpp \
	test/test_properties.cpp \
	$(FFT_SOURCES)

BENCH_SOURCES := \
	bench/run_all_benchmarks.cpp \
	bench/utils.cpp \
	bench/implementations.cpp \
	bench/bench_sizes.cpp \
	bench/bench_steady_state.cpp \
	bench/fftw_wrapper.cpp \
	$(FFT_SOURCES)

.PHONY: all test bench clean memcheck

all: test

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

memcheck: $(BENCH_BIN)
	/usr/bin/time -l ./$(BENCH_BIN)
	/usr/bin/leaks --atExit -- ./$(BENCH_BIN)

$(TEST_BIN): $(TEST_SOURCES)
	mkdir -p build
	$(CXX) $(TEST_CXXFLAGS) $(TEST_SOURCES) -o $(TEST_BIN)

$(BENCH_BIN): $(BENCH_SOURCES)
	mkdir -p build
	$(CXX) $(BENCH_CXXFLAGS) $(BENCH_SOURCES) -o $(BENCH_BIN) $(FFTW_LIBS)

clean:
	rm -rf build
