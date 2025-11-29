#!/bin/bash
# Automated Groq LPU Deployment Pipeline
# Compiles Fortran → MLIR → Groq Binary
# Author: Generated for 3.5-bit LLaMA 70B project
# Date: 2025-11-28

set -e  # Exit on error

echo "=========================================="
echo "Groq LPU Deployment Pipeline"
echo "3.5-bit Quantized LLaMA 70B Inference"
echo "=========================================="
echo ""

# Configuration
FORTRAN_SOURCE="matmul_simd_optimized.f90"
MLIR_OUTPUT_DIR="mlir_output"
GROQ_BINARY_DIR="groq_binaries"

# Create output directories
mkdir -p "$MLIR_OUTPUT_DIR"
mkdir -p "$GROQ_BINARY_DIR"

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
if ! command -v lfortran &> /dev/null; then
    echo "  ⚠ LFortran not found. Install with:"
    echo "    conda install -c conda-forge lfortran"
    echo "  Continuing with example MLIR..."
    USE_EXAMPLE_MLIR=true
else
    echo "  ✓ LFortran found: $(lfortran --version | head -n1)"
    USE_EXAMPLE_MLIR=false
fi

if ! command -v mlir-opt &> /dev/null; then
    echo "  ⚠ mlir-opt not found. Install LLVM/MLIR tools."
    SKIP_MLIR_OPT=true
else
    echo "  ✓ mlir-opt found"
    SKIP_MLIR_OPT=false
fi

if ! command -v groq-compiler &> /dev/null; then
    echo "  ⚠ groq-compiler not found. Requires Groq SDK."
    echo "  Visit: https://console.groq.com/"
    SKIP_GROQ_COMPILE=true
else
    echo "  ✓ groq-compiler found"
    SKIP_GROQ_COMPILE=false
fi

echo ""

# Step 2: Generate MLIR from Fortran
echo "Step 2: Generating MLIR from Fortran..."
if [ "$USE_EXAMPLE_MLIR" = true ]; then
    echo "  Using example MLIR (LFortran not available)"
    cp "$MLIR_OUTPUT_DIR/matmul_int4_groq_example.mlir" "$MLIR_OUTPUT_DIR/matmul_generated.mlir"
else
    echo "  Compiling: $FORTRAN_SOURCE"
    lfortran --show-mlir "$FORTRAN_SOURCE" > "$MLIR_OUTPUT_DIR/matmul_generated.mlir"
    echo "  ✓ MLIR generated: $MLIR_OUTPUT_DIR/matmul_generated.mlir"
fi
echo ""

# Step 3: Optimize MLIR
if [ "$SKIP_MLIR_OPT" = false ]; then
    echo "Step 3: Optimizing MLIR..."

    # Pass 1: Affine optimizations
    echo "  Pass 1: Affine loop transformations..."
    mlir-opt \
      --affine-loop-tile="tile-size=64" \
      --affine-loop-fusion \
      "$MLIR_OUTPUT_DIR/matmul_generated.mlir" \
      -o "$MLIR_OUTPUT_DIR/matmul_affine.mlir"

    # Pass 2: Vectorization
    echo "  Pass 2: Vectorization (tile-size=8)..."
    mlir-opt \
      --affine-vectorize="virtual-vector-size=8" \
      "$MLIR_OUTPUT_DIR/matmul_affine.mlir" \
      -o "$MLIR_OUTPUT_DIR/matmul_vectorized.mlir"

    # Pass 3: Lower to standard dialect
    echo "  Pass 3: Lowering to standard dialect..."
    mlir-opt \
      --lower-affine \
      --convert-scf-to-cf \
      "$MLIR_OUTPUT_DIR/matmul_vectorized.mlir" \
      -o "$MLIR_OUTPUT_DIR/matmul_lowered.mlir"

    echo "  ✓ MLIR optimization complete"
    MLIR_INPUT="$MLIR_OUTPUT_DIR/matmul_lowered.mlir"
else
    echo "Step 3: Skipping MLIR optimization (mlir-opt not available)"
    MLIR_INPUT="$MLIR_OUTPUT_DIR/matmul_generated.mlir"
fi
echo ""

# Step 4: Compile to Groq binary
if [ "$SKIP_GROQ_COMPILE" = false ]; then
    echo "Step 4: Compiling to Groq LPU binary..."

    groq-compiler \
      --target=lpu \
      --optimization-level=3 \
      --enable-systolic-array \
      --enable-tensor-cores \
      --tile-size=320 \
      "$MLIR_INPUT" \
      -o "$GROQ_BINARY_DIR/llama70b_3p5bit.lpubin"

    echo "  ✓ Groq binary created: $GROQ_BINARY_DIR/llama70b_3p5bit.lpubin"

    # Generate performance report
    echo "  Analyzing performance..."
    groq-compiler \
      --target=lpu \
      --analyze-performance \
      --report-file="$GROQ_BINARY_DIR/performance_report.json" \
      "$MLIR_INPUT"

    echo "  ✓ Performance report: $GROQ_BINARY_DIR/performance_report.json"
else
    echo "Step 4: Skipping Groq compilation (groq-compiler not available)"
    echo "  Install Groq SDK: https://console.groq.com/"
fi
echo ""

# Step 5: Summary
echo "=========================================="
echo "Deployment Pipeline Summary"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  MLIR:          $MLIR_INPUT"
if [ "$SKIP_GROQ_COMPILE" = false ]; then
echo "  Groq Binary:   $GROQ_BINARY_DIR/llama70b_3p5bit.lpubin"
echo "  Perf Report:   $GROQ_BINARY_DIR/performance_report.json"
fi
echo ""
echo "Next steps:"
if [ "$SKIP_GROQ_COMPILE" = false ]; then
echo "  1. Upload binary: groq-cli upload --binary $GROQ_BINARY_DIR/llama70b_3p5bit.lpubin"
echo "  2. Run inference: groq-cli run --binary llama70b_3p5bit.lpubin"
echo "  3. Benchmark:     groq-cli benchmark --binary llama70b_3p5bit.lpubin --iterations 1000"
else
echo "  1. Install Groq SDK and groq-compiler"
echo "  2. Rerun this script to complete compilation"
fi
echo ""
echo "✓ Pipeline complete!"
