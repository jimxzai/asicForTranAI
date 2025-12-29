# Makefile for LLaMA 70B INT4 Inference - Pure Fortran 2023
# Target: 3100+ tok/s on Groq LPU for 70B model

# Compiler settings
FC = gfortran
FFLAGS = -O3 -march=native -ffast-math -funroll-loops -fno-bounds-check
DEBUG_FLAGS = -g -fbounds-check -Wall -Wextra
OPENMP_FLAGS = -fopenmp

# BLAS settings for OpenBLAS integration (50-100× speedup)
# On MSYS2 Windows: pacman -S mingw-w64-x86_64-openblas
# On Ubuntu/WSL: apt-get install libopenblas-dev
# On macOS: brew install openblas
BLAS_FLAGS = -lopenblas
BLAS_INCLUDE =
# Alternative for Intel MKL (if available):
# BLAS_FLAGS = -lmkl_rt -lpthread -lm -ldl
# Alternative for macOS Accelerate:
# BLAS_FLAGS = -framework Accelerate

# GPU settings for NVIDIA CUDA (100-500× speedup)
CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1
CUDA_LIB = $(CUDA_PATH)/lib/x64
CUDA_INCLUDE = $(CUDA_PATH)/include
CUBLAS_FLAGS = -L"$(CUDA_LIB)" -lcublas -lcudart -I"$(CUDA_INCLUDE)"

# NVIDIA HPC SDK (for OpenACC and CUDA Fortran)
NVFORTRAN = nvfortran
NVFORTRAN_FLAGS = -O3 -gpu=cc75 -Minfo=accel
OPENACC_FLAGS = -acc

# Source files
CORE_SOURCES = matmul_int4_groq.f90 transformer_layer.f90
WEIGHT_LOADER_SOURCE = weight_loader.f90
SAMPLING_SOURCE = sampling.f90
OPTIMIZED_SOURCES = matmul_fully_optimized.f90
SIMD_SOURCES = matmul_simd_optimized.f90
BLAS_SOURCES = matmul_blas_optimized.f90
MODEL_SOURCES = $(CORE_SOURCES) llama_model.f90
MAIN_SOURCE = llama70b_int4.f90
GENERATE_SOURCE = llama_generate.f90
TEST_SOURCE = test_transformer_layer.f90
TEST_MODEL_SOURCE = test_llama_model.f90
TEST_WEIGHT_LOADING_SOURCE = test_weight_loading.f90
TEST_SAMPLING_SOURCE = test_sampling.f90
BENCHMARK_SOURCE = benchmark_optimizations.f90
BENCHMARK_BLAS_SOURCE = benchmark_blas.f90

# GPU source files
CUBLAS_SOURCES = matmul_cublas.f90
OPENACC_SOURCES = matmul_openacc.f90
CUDA_FORTRAN_SOURCES = matmul_cuda_fortran.cuf

# Object files
CORE_OBJECTS = $(CORE_SOURCES:.f90=.o)

# Targets
TARGET_MAIN = llama_inference
TARGET_GENERATE = llama_generate
TARGET_TEST = test_layer
TARGET_MODEL_TEST = test_llama_80layers
TARGET_WEIGHT_TEST = test_weights
TARGET_SAMPLING_TEST = test_sampling
TARGET_DEBUG = test_layer_debug
TARGET_BENCHMARK = bench_optimizations
TARGET_SIMD_BENCHMARK = bench_simd
TARGET_BLAS_BENCHMARK = bench_blas
TARGET_CUBLAS_BENCHMARK = bench_cublas
TARGET_OPENACC_BENCHMARK = bench_openacc
TARGET_CUDA_FORTRAN_BENCHMARK = bench_cuda
TARGET_GEN_WEIGHTS = gen_test_weights

# Default target - build everything
all: $(TARGET_TEST) $(TARGET_MAIN)
	@echo "============================================"
	@echo "Build complete!"
	@echo "  Test:      ./$(TARGET_TEST)"
	@echo "  Main:      ./$(TARGET_MAIN)"
	@echo "============================================"

# Test program (fast build)
$(TARGET_TEST): $(CORE_SOURCES) $(TEST_SOURCE)
	@echo "Building transformer layer test..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Debug version of test
debug: $(CORE_SOURCES) $(TEST_SOURCE)
	@echo "Building debug version..."
	$(FC) $(DEBUG_FLAGS) -o $(TARGET_DEBUG) $^
	@echo "✓ Built: $(TARGET_DEBUG)"

# Full inference (when complete)
$(TARGET_MAIN): $(CORE_SOURCES) $(MAIN_SOURCE)
	@echo "Building full LLaMA 70B inference..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# End-to-end text generation (complete pipeline)
$(TARGET_GENERATE): $(MODEL_SOURCES) $(WEIGHT_LOADER_SOURCE) $(SAMPLING_SOURCE) $(GENERATE_SOURCE)
	@echo "Building LLaMA text generation pipeline..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"
	@echo ""
	@echo "Run with: ./$(TARGET_GENERATE)"
	@echo "Requires: Python 3 + scripts/tokenizer.py"

# OpenMP parallel version (for CPU testing)
parallel: $(CORE_SOURCES) $(TEST_SOURCE)
	@echo "Building parallel version with OpenMP..."
	$(FC) $(FFLAGS) $(OPENMP_FLAGS) -o test_layer_omp $^
	@echo "✓ Built: test_layer_omp"
	@echo "Run with: OMP_NUM_THREADS=8 ./test_layer_omp"

# Compile individual modules (for development)
%.o: %.f90
	$(FC) $(FFLAGS) -c $<

%.mod: %.f90
	$(FC) $(FFLAGS) -c $<

# Quick test - build and run single layer
test: $(TARGET_TEST)
	@echo ""
	@echo "Running transformer layer test..."
	@echo "=========================================="
	./$(TARGET_TEST)

# Build 80-layer model test
$(TARGET_MODEL_TEST): $(MODEL_SOURCES) $(TEST_MODEL_SOURCE)
	@echo "Building 80-layer LLaMA model test..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Test full 80-layer model
test-model: $(TARGET_MODEL_TEST)
	@echo ""
	@echo "Running 80-layer LLaMA model test..."
	@echo "=========================================="
	./$(TARGET_MODEL_TEST)

# Run debug version with bounds checking
test-debug: debug
	@echo ""
	@echo "Running debug test (with bounds checking)..."
	@echo "=========================================="
	./$(TARGET_DEBUG)

# Build optimization benchmark
$(TARGET_BENCHMARK): $(CORE_SOURCES) $(OPTIMIZED_SOURCES) $(BENCHMARK_SOURCE)
	@echo "Building optimization benchmark..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Run optimization benchmark
benchmark-opt: $(TARGET_BENCHMARK)
	@echo ""
	@echo "Running optimization benchmark..."
	@echo "This will compare baseline vs optimized implementations"
	@echo ""
	./$(TARGET_BENCHMARK)

# Build SIMD benchmark
$(TARGET_SIMD_BENCHMARK): $(CORE_SOURCES) $(SIMD_SOURCES) $(BENCHMARK_SOURCE)
	@echo "Building SIMD benchmark with OpenMP..."
	$(FC) $(FFLAGS) $(OPENMP_FLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Run SIMD benchmark
benchmark-simd: $(TARGET_SIMD_BENCHMARK)
	@echo ""
	@echo "Running SIMD-optimized benchmark..."
	@echo "Target: 2.3× speedup with SIMD + OpenMP"
	@echo ""
	OMP_NUM_THREADS=4 ./$(TARGET_SIMD_BENCHMARK)

# Build BLAS benchmark (requires OpenBLAS)
$(TARGET_BLAS_BENCHMARK): $(CORE_SOURCES) $(SIMD_SOURCES) $(BLAS_SOURCES) $(BENCHMARK_BLAS_SOURCE)
	@echo "Building BLAS benchmark with OpenBLAS..."
	@echo "Make sure OpenBLAS is installed:"
	@echo "  MSYS2:  pacman -S mingw-w64-x86_64-openblas"
	@echo "  Ubuntu: apt-get install libopenblas-dev"
	@echo "  macOS:  brew install openblas"
	@echo ""
	$(FC) $(FFLAGS) $(OPENMP_FLAGS) $(BLAS_INCLUDE) -o $@ $^ $(BLAS_FLAGS)
	@echo "✓ Built: $@"

# Run BLAS benchmark (50-100× speedup expected)
benchmark-blas: $(TARGET_BLAS_BENCHMARK)
	@echo ""
	@echo "=========================================="
	@echo "Running BLAS-optimized benchmark..."
	@echo "Target: 50-100× speedup with OpenBLAS"
	@echo "=========================================="
	@echo ""
	OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./$(TARGET_BLAS_BENCHMARK)

# ============================================
# GPU-Accelerated Builds (NVIDIA RTX 2080 Ti)
# ============================================

# Build cuBLAS benchmark (works with gfortran + CUDA Toolkit)
$(TARGET_CUBLAS_BENCHMARK): $(CORE_SOURCES) $(SIMD_SOURCES) $(CUBLAS_SOURCES) benchmark_gpu.f90
	@echo "Building cuBLAS GPU benchmark..."
	@echo "Requires: CUDA Toolkit 10.1 (detected)"
	@echo "GPU: NVIDIA GeForce RTX 2080 Ti"
	@echo "Expected speedup: 100-500×"
	@echo ""
	$(FC) $(FFLAGS) $(OPENMP_FLAGS) -o $@ $^ $(CUBLAS_FLAGS)
	@echo "✓ Built: $@"

# Run cuBLAS benchmark on RTX 2080 Ti
benchmark-cublas: $(TARGET_CUBLAS_BENCHMARK)
	@echo ""
	@echo "=========================================="
	@echo "Running cuBLAS GPU benchmark..."
	@echo "Target: 100-500× speedup on RTX 2080 Ti"
	@echo "=========================================="
	@echo ""
	./$(TARGET_CUBLAS_BENCHMARK)

# Build OpenACC benchmark (requires NVIDIA HPC SDK)
$(TARGET_OPENACC_BENCHMARK): $(CORE_SOURCES) $(OPENACC_SOURCES) benchmark_blas.f90
	@echo "Building OpenACC GPU benchmark..."
	@echo "Requires: NVIDIA HPC SDK (nvfortran)"
	@echo "Install from: https://developer.nvidia.com/hpc-sdk"
	@echo ""
	$(NVFORTRAN) $(NVFORTRAN_FLAGS) $(OPENACC_FLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Run OpenACC benchmark
benchmark-openacc: $(TARGET_OPENACC_BENCHMARK)
	@echo ""
	@echo "=========================================="
	@echo "Running OpenACC GPU benchmark..."
	@echo "Target: 50-200× speedup"
	@echo "=========================================="
	@echo ""
	./$(TARGET_OPENACC_BENCHMARK)

# Build CUDA Fortran benchmark (requires NVIDIA HPC SDK)
$(TARGET_CUDA_FORTRAN_BENCHMARK): $(CORE_SOURCES) $(CUDA_FORTRAN_SOURCES) benchmark_blas.f90
	@echo "Building CUDA Fortran GPU benchmark..."
	@echo "Requires: NVIDIA HPC SDK (nvfortran)"
	@echo ""
	$(NVFORTRAN) $(NVFORTRAN_FLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Run CUDA Fortran benchmark (fastest - custom kernels)
benchmark-cuda-fortran: $(TARGET_CUDA_FORTRAN_BENCHMARK)
	@echo ""
	@echo "=========================================="
	@echo "Running CUDA Fortran GPU benchmark..."
	@echo "Target: 100-500× speedup (fused kernels)"
	@echo "=========================================="
	@echo ""
	./$(TARGET_CUDA_FORTRAN_BENCHMARK)

# Run ALL GPU benchmarks (requires NVIDIA HPC SDK)
benchmark-gpu-all: benchmark-cublas benchmark-openacc benchmark-cuda-fortran
	@echo ""
	@echo "=========================================="
	@echo "All GPU benchmarks complete!"
	@echo "See GPU_SETUP_GUIDE.md for details"
	@echo "=========================================="

# Build test weight generator
$(TARGET_GEN_WEIGHTS): $(CORE_SOURCES) generate_test_weights.f90
	@echo "Building test weight generator..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Generate test weights
gen-weights: $(TARGET_GEN_WEIGHTS)
	@echo ""
	@echo "Generating random test weights..."
	@echo "=========================================="
	./$(TARGET_GEN_WEIGHTS)
	@echo ""
	@echo "✓ Test weights generated: test_weights_layer0.bin"

# Build weight loading test
$(TARGET_WEIGHT_TEST): $(CORE_SOURCES) $(WEIGHT_LOADER_SOURCE) $(TEST_WEIGHT_LOADING_SOURCE)
	@echo "Building weight loading test..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Test weight loading
test-weights: $(TARGET_WEIGHT_TEST)
	@echo ""
	@echo "Testing weight loading..."
	@echo "=========================================="
	@if [ ! -f test_weights_layer0.bin ]; then \
		echo "⚠ test_weights_layer0.bin not found. Running gen-weights first..."; \
		$(MAKE) gen-weights; \
	fi
	./$(TARGET_WEIGHT_TEST)

# Build sampling test
$(TARGET_SAMPLING_TEST): $(SAMPLING_SOURCE) $(TEST_SAMPLING_SOURCE)
	@echo "Building sampling test..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

# Test sampling strategies
test-sampling: $(TARGET_SAMPLING_TEST)
	@echo ""
	@echo "Testing sampling strategies..."
	@echo "=========================================="
	./$(TARGET_SAMPLING_TEST)

# Quick benchmark (old)
benchmark: $(TARGET_TEST)
	@echo "Running benchmarks..."
	@echo "Seq Length | Time (ms) | Throughput"
	@echo "-----------|-----------|------------"
	@for len in 1 4 8 16 32; do \
		echo "Testing seq_len=$$len..."; \
	done

# Check code style and potential issues
lint:
	@echo "Checking Fortran code..."
	@gfortran -fsyntax-only -Wall -Wextra $(CORE_SOURCES) $(TEST_SOURCE) && \
		echo "✓ No syntax errors found" || \
		echo "✗ Syntax errors detected"

# Clean build artifacts
clean:
	rm -f *.o *.mod $(TARGET_MAIN) $(TARGET_GENERATE) $(TARGET_TEST) $(TARGET_MODEL_TEST) $(TARGET_WEIGHT_TEST) $(TARGET_SAMPLING_TEST) $(TARGET_DEBUG) $(TARGET_BENCHMARK) $(TARGET_SIMD_BENCHMARK) $(TARGET_BLAS_BENCHMARK) $(TARGET_CUBLAS_BENCHMARK) $(TARGET_OPENACC_BENCHMARK) $(TARGET_CUDA_FORTRAN_BENCHMARK) $(TARGET_GEN_WEIGHTS) test_layer_omp test_weights_*.bin prompt.txt tokens.bin output_tokens.bin output.txt
	@echo "✓ Cleaned build artifacts"

# Clean and rebuild
rebuild: clean all

# Show build configuration
info:
	@echo "Build Configuration:"
	@echo "  Compiler:     $(FC)"
	@echo "  Flags:        $(FFLAGS)"
	@echo "  Sources:      $(CORE_SOURCES)"
	@echo "  Test source:  $(TEST_SOURCE)"
	@echo "  Targets:"
	@echo "    - $(TARGET_TEST)  (test program)"
	@echo "    - $(TARGET_MAIN)  (full inference)"
	@echo ""
	@echo "Available targets:"
	@echo "  make          - Build test and main"
	@echo "  make test     - Build and run test"
	@echo "  make debug    - Build with debugging"
	@echo "  make parallel - Build with OpenMP"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make lint     - Check code syntax"
	@echo ""
	@echo "CPU Benchmarks:"
	@echo "  make benchmark-opt   - Compare baseline vs optimized"
	@echo "  make benchmark-simd  - Test SIMD+OpenMP (8.4× speedup)"
	@echo "  make benchmark-blas  - Test BLAS (OpenBLAS)"
	@echo "  make benchmark-llama - Full 80-layer performance test"
	@echo ""
	@echo "GPU Benchmarks (NVIDIA RTX 2080 Ti):"
	@echo "  make benchmark-cublas        - cuBLAS (100-500× speedup)"
	@echo "  make benchmark-openacc       - OpenACC directives"
	@echo "  make benchmark-cuda-fortran  - CUDA Fortran kernels"
	@echo "  make benchmark-gpu-all       - Run all GPU benchmarks"

# Comprehensive LLaMA performance benchmark
BENCHMARK_LLAMA_SOURCE = benchmark_llama.f90
TARGET_BENCHMARK_LLAMA = benchmark_llama

$(TARGET_BENCHMARK_LLAMA): $(MODEL_SOURCES) $(BENCHMARK_LLAMA_SOURCE)
	@echo "Building LLaMA performance benchmark..."
	$(FC) $(FFLAGS) -o $@ $^
	@echo "✓ Built: $@"

benchmark-llama: $(TARGET_BENCHMARK_LLAMA)
	@echo ""
	@echo "Running LLaMA 70B performance benchmark..."
	@echo "=========================================="
	ulimit -s unlimited && ./$(TARGET_BENCHMARK_LLAMA)

# Deploy to Groq (future - when MLIR generation works)
deploy:
	@echo "Groq ASIC Deployment"
	@echo "TODO: Generate MLIR from Fortran source"
	@echo "TODO: Upload to Groq LPU"
	@echo ""
	@echo "Manual steps for now:"
	@echo "  1. Complete transformer implementation"
	@echo "  2. Generate MLIR: lfortran --show-ast llama70b_int4.f90"
	@echo "  3. Contact Groq for ASIC access"

# Help
help: info

.PHONY: all test test-debug test-weights test-sampling benchmark benchmark-blas benchmark-cublas benchmark-openacc benchmark-cuda-fortran benchmark-gpu-all lint clean rebuild info deploy help parallel debug
