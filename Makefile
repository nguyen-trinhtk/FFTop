# Override on the command line: make CXX=g++-15
CXX ?= g++
ifneq ($(wildcard /opt/homebrew/bin/g++-15),)
CXX := /opt/homebrew/bin/g++-15
endif

NVCC ?= nvcc
# Colab T4: sm_75. Override: make CUDA_ARCH=sm_80
CUDA_ARCH ?= sm_75

INCLUDES := -Iinclude
CXXFLAGS := -g -O3 -march=native -std=c++17 -Wall -Wextra $(INCLUDES)
OPENMP_FLAGS := -fopenmp

FFTW_CFLAGS := $(shell pkg-config --cflags fftw3 2>/dev/null)
FFTW_LIBS := $(shell pkg-config --libs fftw3 2>/dev/null)
ifeq ($(FFTW_LIBS),)
FFTW_CFLAGS := -I/opt/homebrew/include
FFTW_LIBS := -L/opt/homebrew/lib -lfftw3
endif

HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

# --- sources -----------------------------------------------------------------

REF_SOURCES := src/fft/ref/dft.cpp

CPU_SOURCES := \
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

GPU_SOURCES := \
	src/fft/gpu/detail/iterative.cu \
	src/fft/gpu/naive.cu \
	src/fft/gpu/optimized.cu

TEST_HARNESS := \
	test/run_all_tests.cpp \
	test/utils.cpp \
	test/test_edge_cases.cpp \
	test/test_power_of_2.cpp \
	test/test_properties.cpp

BENCH_HARNESS := \
	bench/run_all_benchmarks.cpp \
	bench/utils.cpp \
	bench/bench_sizes.cpp \
	bench/bench_steady_state.cpp

TEST_CPU_SOURCES := \
	$(TEST_HARNESS) \
	test/cpu_implementations.cpp \
	$(REF_SOURCES) \
	$(CPU_SOURCES)

TEST_GPU_SOURCES := \
	$(TEST_HARNESS) \
	test/gpu_implementations.cpp \
	$(REF_SOURCES) \
	$(GPU_SOURCES)

BENCH_CPU_SOURCES := \
	$(BENCH_HARNESS) \
	bench/cpu_implementations.cpp \
	bench/fftw_wrapper.cpp \
	$(CPU_SOURCES)

BENCH_GPU_SOURCES := \
	$(BENCH_HARNESS) \
	bench/gpu_implementations.cpp \
	$(GPU_SOURCES)

# --- binaries ----------------------------------------------------------------

TEST_CPU_BIN := build/run_cpu_tests
TEST_GPU_BIN := build/run_gpu_tests
BENCH_CPU_BIN := build/run_cpu_benchmarks
BENCH_GPU_BIN := build/run_gpu_benchmarks

# --- targets -----------------------------------------------------------------

.PHONY: all test test-cpu test-gpu bench bench-cpu bench-gpu clean memcheck

all: test-cpu

# Aliases: default local path is CPU.
test: test-cpu
bench: bench-cpu

test-cpu: $(TEST_CPU_BIN)
	./$(TEST_CPU_BIN)

bench-cpu: $(BENCH_CPU_BIN)
	./$(BENCH_CPU_BIN)

memcheck: $(BENCH_CPU_BIN)
	/usr/bin/time -l ./$(BENCH_CPU_BIN)
	/usr/bin/leaks --atExit -- ./$(BENCH_CPU_BIN)

$(TEST_CPU_BIN): $(TEST_CPU_SOURCES)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(TEST_CPU_SOURCES) -o $(TEST_CPU_BIN)

$(BENCH_CPU_BIN): $(BENCH_CPU_SOURCES)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(FFTW_CFLAGS) $(OPENMP_FLAGS) $(BENCH_CPU_SOURCES) \
		-o $(BENCH_CPU_BIN) $(FFTW_LIBS)

# GPU (requires nvcc). On Colab T4: make test-gpu bench-gpu CUDA_ARCH=sm_75
ifeq ($(HAVE_NVCC),)
test-gpu bench-gpu:
	@echo "nvcc not found — use Google Colab (T4) or a CUDA machine."
	@exit 1
else
test-gpu: $(TEST_GPU_BIN)
	./$(TEST_GPU_BIN)

bench-gpu: $(BENCH_GPU_BIN)
	./$(BENCH_GPU_BIN)

$(TEST_GPU_BIN): $(TEST_GPU_SOURCES)
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) $(INCLUDES) \
		$(TEST_GPU_SOURCES) -o $(TEST_GPU_BIN) -lcudart

$(BENCH_GPU_BIN): $(BENCH_GPU_SOURCES)
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) $(INCLUDES) \
		$(BENCH_GPU_SOURCES) -o $(BENCH_GPU_BIN) -lcudart
endif

clean:
	rm -rf build
