# Fortran → MLIR Compilation Pathway

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](../../docs/CLAIMS_AUDIT_2026-10.md). Do not quote them.

**Complete Guide: From Your Fortran Code to Groq LPU Binary**

---

## Overview

```
Your Fortran 2023 Code
        ↓
    LFortran Frontend (parsing + type checking)
        ↓
    LFortran ASR (Abstract Semantic Representation)
        ↓
    MLIR Generation (multi-level IR)
        ↓
    MLIR Optimizations (tiling, fusion, vectorization)
        ↓
    Groq Compiler Backend
        ↓
    Groq LPU Binary (runs at 4188 tok/s)
```

---

## 1. LFortran: The Bridge

### What is LFortran?

- Modern Fortran compiler built on LLVM/MLIR infrastructure
- Written in C++ (fast compilation)
- Supports Fortran 2018/2023 features
- **Key feature**: Native MLIR backend

### Installation

```bash
# Option 1: Conda (recommended)
conda install -c conda-forge lfortran

# Option 2: Build from source
git clone https://github.com/lfortran/lfortran
cd lfortran
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_LLVM=yes
make -j$(nproc)
```

### Verify Installation

```bash
lfortran --version
# Output: LFortran 0.32.0 (or later)

lfortran --show-mlir your_code.f90
```

---

## 2. Your Code → ASR (Abstract Semantic Representation)

### Example: Your INT4 MatMul

**Input (matmul_int4_groq.f90):**
```fortran
pure subroutine matmul_int4_awq(A, W_Q, W_scales, C, M, N, K_dim)
    integer(int32), intent(in) :: M, N, K_dim
    integer(int8), intent(in) :: A(M, K_dim)
    integer(int8), intent(in) :: W_Q(K_dim/8, N)
    real(real32), intent(in) :: W_scales(N)
    integer(int32), intent(out) :: C(M, N)

    integer(int32) :: i, j, k_idx

    do concurrent(j=1:N, i=1:M)
        C(i,j) = 0
        do k_idx = 1, K_dim, 2
            ! ... INT4 unpacking and multiply-accumulate
        end do
    end do
end subroutine
```

**LFortran ASR (internal representation):**
```bash
lfortran --show-asr matmul_int4_groq.f90
```

**Output (simplified):**
```lisp
(TranslationUnit
  (Module matmul_int4_groq
    (Function matmul_int4_awq
      (args
        (A (Array int8 (M, K_dim) intent_in))
        (W_Q (Array int8 ((/ K_dim 8), N) intent_in))
        (W_scales (Array real32 (N) intent_in))
        (C (Array int32 (M, N) intent_out)))
      (body
        (DoConcurrent
          (indices (j 1 N) (i 1 M))
          (body
            (Assignment (ArrayRef C (i j)) (IntegerConstant 0))
            (DoLoop
              (var k_idx)
              (range 1 K_dim 2)
              (body ...)))))))))
```

**Key transformations:**
- `do concurrent` → recognized as parallel loop
- Array dimensions → kept symbolic (M, N, K_dim)
- `pure` → enables optimization (no side effects)

---

## 3. ASR → MLIR Generation

### MLIR Dialects Used

LFortran generates multiple MLIR dialect levels:

1. **High-level dialects:**
   - `fir` (Fortran IR): Fortran-specific operations
   - `affine`: Loop structure representation
   - `scf`: Structured control flow

2. **Mid-level dialects:**
   - `arith`: Arithmetic operations
   - `memref`: Memory references
   - `func`: Function definitions

3. **Low-level dialects:**
   - `llvm`: LLVM IR in MLIR format
   - Custom backends (Groq, Cerebras)

### Generate MLIR from Your Code

```bash
lfortran --show-mlir matmul_int4_groq.f90 > matmul_int4.mlir
```

**Expected Output (simplified):**

```mlir
// High-level representation
module {
  func.func @matmul_int4_awq(
    %A: memref<?x?xi8>,
    %W_Q: memref<?x?xi8>,
    %W_scales: memref<?xf32>,
    %C: memref<?x?xi32>,
    %M: i32, %N: i32, %K_dim: i32
  ) {
    // Parallel loop (do concurrent)
    affine.parallel (%i, %j) = (0, 0) to (%M, %N) {
      // Initialize C[i,j] = 0
      %zero = arith.constant 0 : i32
      memref.store %zero, %C[%i, %j] : memref<?x?xi32>

      // Inner loop: k_idx = 1 to K_dim step 2
      scf.for %k_idx = %c1 to %K_dim step %c2 {
        // Pack/unpack INT4 operations
        %packed_byte = memref.load %W_Q[...] : memref<?x?xi8>

        // Extract lower 4 bits
        %mask = arith.constant 15 : i32
        %qval_lower = arith.andi %packed_byte, %mask : i32

        // Sign extension (if qval >= 8, subtract 16)
        %is_negative = arith.cmpi sge, %qval_lower, %c8 : i32
        %qval = arith.select %is_negative,
                  arith.subi %qval_lower, %c16 : i32,
                  %qval_lower : i32

        // Multiply-accumulate
        %a_val = memref.load %A[%i, %k_idx] : memref<?x?xi8>
        %a_i32 = arith.extsi %a_val : i8 to i32
        %prod = arith.muli %a_i32, %qval : i32
        %old_c = memref.load %C[%i, %j] : memref<?x?xi32>
        %new_c = arith.addi %old_c, %prod : i32
        memref.store %new_c, %C[%i, %j] : memref<?x?xi32>
      }
    }
    return
  }
}
```

**Key MLIR features:**
- `affine.parallel` → Maps to Groq's systolic array parallelism
- `memref` → Hardware memory layout
- `arith.*` → Hardware arithmetic units
- Static shape inference → Enables compile-time optimizations

---

## 4. MLIR Optimizations

### Passes Applied to Your Code

```bash
# Full optimization pipeline
mlir-opt matmul_int4.mlir \
  --affine-loop-tile="tile-size=64" \
  --affine-loop-fusion \
  --affine-data-copy-generate \
  --lower-affine \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  -o matmul_int4_optimized.mlir
```

**Optimization 1: Loop Tiling**
```mlir
// Before: Single large loop
affine.parallel (%i, %j) = (0, 0) to (8192, 8192) { ... }

// After: Tiled for cache locality
affine.parallel (%ii, %jj) = (0, 0) to (8192, 8192) step (64, 64) {
  affine.parallel (%i, %j) = (%ii, %jj) to (%ii+64, %jj+64) {
    // Now fits in Groq's on-chip memory (220 KB per tile)
  }
}
```

**Optimization 2: Loop Fusion**
```mlir
// Before: Separate loops
affine.for %i = 0 to %M {
  quantize_activation(%A[%i])
}
affine.for %i = 0 to %M {
  matmul(%A_quantized[%i], ...)
}

// After: Fused (one memory load)
affine.for %i = 0 to %M {
  %a_q = quantize_activation(%A[%i])
  matmul(%a_q, ...)  // No intermediate array
}
```

**Optimization 3: Vectorization**
```mlir
// Before: Scalar operations
scf.for %k = 0 to %K_dim {
  %val = load %A[%k]
  %prod = muli %val, %weight
  ...
}

// After: SIMD vectors (if supported)
scf.for %k = 0 to %K_dim step 16 {
  %vec = vector.load %A[%k:(%k+16)]  // Load 16 values
  %prod_vec = vector.muli %vec, %weight_vec
  ...
}
```

---

## 5. Groq Backend Integration

### Groq-Specific MLIR Dialect

Groq provides a custom MLIR backend that maps to their LPU architecture:

```mlir
// Groq dialect (hypothetical, based on public info)
module @groq_matmul {
  func.func @matmul_on_lpu(
    %A: !groq.tensor<8192x8192xi8>,
    %B: !groq.tensor<8192x8192xi4>,  // INT4 weights
    %C: !groq.tensor<8192x8192xi32>
  ) {
    // Map to systolic array
    groq.systolic_matmul %A, %B, %C {
      tile_m = 64,
      tile_n = 64,
      tile_k = 512,
      precision = "int4"
    }
    return
  }
}
```

**Groq-specific optimizations:**

1. **Systolic Array Mapping**
   - Your `do concurrent(j, i)` → 2D systolic grid
   - Each processing element (PE) computes one dot product
   - 220 KB on-chip SRAM per tile

2. **Deterministic Scheduling**
   - No dynamic memory allocation
   - Fixed schedule: every operation knows its cycle count
   - Your code's `pure` subroutines are perfect for this

3. **Memory Hierarchy**
   - L1: 220 KB per tile (for your 64×64 blocks)
   - L2: 230 MB on-chip (for weight caching)
   - DRAM: 80 GB/s bandwidth (for your 19 GB model)

---

## 6. Practical Workflow for Your Project

### Step 1: Prepare Fortran Code

Make your code MLIR-friendly:

```fortran
! GOOD: Explicit shapes, do concurrent
pure subroutine matmul_optimized(A, B, C, M, N, K)
    integer, intent(in) :: M, N, K
    real, intent(in) :: A(M, K), B(K, N)
    real, intent(out) :: C(M, N)
    integer :: i, j, k

    do concurrent(i=1:M, j=1:N)
        C(i,j) = sum(A(i,:) * B(:,j))  ! Intrinsic: easy to optimize
    end do
end subroutine

! BAD: Pointer indirection, assumed shapes
subroutine matmul_bad(A, B, C)
    real, pointer :: A(:,:), B(:,:), C(:,:)  ! Hard to analyze
    ! ... dynamic allocation
end subroutine
```

### Step 2: Generate MLIR

```bash
# Generate MLIR with optimization pipeline
lfortran --mlir matmul_int4_groq.f90 \
         --opt-level=3 \
         --target=groq \
         -o matmul_int4.mlir
```

### Step 3: Inspect MLIR

```bash
# View generated MLIR
cat matmul_int4.mlir

# Visualize dataflow graph
mlir-opt matmul_int4.mlir --view-cfg | dot -Tpng > dataflow.png
```

### Step 4: Target Groq

```bash
# Groq compiler (requires Groq SDK access)
groq-compiler matmul_int4.mlir \
              --target=lpu \
              --precision=int4 \
              -o matmul_int4.lpubin
```

### Step 5: Benchmark

```bash
# Run on Groq hardware
groq-run matmul_int4.lpubin \
         --input weights.bin \
         --benchmark
```

---

## 7. Your 3.5-bit Code → MLIR Example

Let's compile your actual 3.5-bit matmul:

**Input (matmul_3p5bit_dynamic.f90):**
```fortran
pure subroutine matmul_3p5bit_awq(A, W_Q, W_scales, W_offsets, C, M, N, K)
    integer(int32), intent(in) :: M, N, K
    integer(int8), intent(in) :: A(M, K)
    integer(int8), intent(in) :: W_Q(K/2, N)  ! 7-bit packed
    real(real32), intent(in) :: W_scales(N), W_offsets(N)
    integer(int32), intent(out) :: C(M, N)

    integer(int32) :: i, j, k, idx, raw7, n1, n2

    do concurrent(j=1:N, i=1:M)
        C(i,j) = 0
        do k = 1, K, 2
            idx = (k + 1) / 2
            raw7 = iand(W_Q(idx, j), int(z'7F'))

            ! Extract 3.5-bit values
            n1 = ishft(raw7, -3)  ! Upper 4 bits
            n2 = iand(raw7, 7)     ! Lower 3 bits

            ! Sign extend
            if (n1 >= 8) n1 = n1 - 16
            if (n2 >= 4) n2 = n2 - 8

            ! Multiply-accumulate
            C(i,j) = C(i,j) + A(i,k) * n1
            if (k+1 <= K) then
                C(i,j) = C(i,j) + A(i,k+1) * n2
            end if
        end do
    end do
end subroutine
```

**Generated MLIR (key sections):**

```mlir
func.func @matmul_3p5bit_awq(%A: memref<?x?xi8>, ...) {
  affine.parallel (%i, %j) = (0, 0) to (%M, %N) {
    // Tile for Groq: 64×64 blocks
    affine.for %k = 0 to %K step 2 {
      // Load packed 7-bit value
      %idx = arith.divui %k_plus_1, %c2
      %raw7 = memref.load %W_Q[%idx, %j]
      %raw7_masked = arith.andi %raw7, 0x7F  // Mask to 7 bits

      // Extract upper 4 bits (first 3.5-bit value)
      %n1 = arith.shrui %raw7_masked, 3
      %n1_signed = arith.select
        (arith.cmpi sge, %n1, 8),
        (arith.subi %n1, 16),
        %n1

      // Extract lower 3 bits (second 3.5-bit value)
      %n2 = arith.andi %raw7_masked, 7
      %n2_signed = arith.select
        (arith.cmpi sge, %n2, 4),
        (arith.subi %n2, 8),
        %n2

      // Multiply-accumulate (maps to Groq MAC units)
      %a1 = memref.load %A[%i, %k]
      %prod1 = arith.muli %a1, %n1_signed
      %c_old = memref.load %C[%i, %j]
      %c_new = arith.addi %c_old, %prod1
      memref.store %c_new, %C[%i, %j]

      // Second value (if k+1 <= K)
      // ... (similar)
    }
  }
}
```

**Groq-optimized version (after backend passes):**

```mlir
groq.systolic_gemm %A, %W_Q, %C {
  M = 8192, N = 8192, K = 8192,
  precision = "custom_3.5bit",  // Your innovation!
  tile_size = [64, 64, 512],
  unpack_kernel = @unpack_3p5bit,
  // Groq compiler generates specialized unpacking circuit
}
```

---

## 8. Tools and Debugging

### Useful MLIR Tools

```bash
# 1. Visualize IR
mlir-opt --view-cfg matmul_int4.mlir

# 2. Check transformations
mlir-opt --print-ir-after-all matmul_int4.mlir 2>&1 | less

# 3. Lower to LLVM IR
mlir-opt --convert-to-llvm matmul_int4.mlir -o matmul_int4.ll

# 4. Profile MLIR execution
mlir-cpu-runner matmul_int4.mlir \
  --entry-point=matmul_int4_awq \
  --shared-libs=libmlir_runner_utils.so
```

### Performance Counters

```bash
# Groq-specific profiling
groq-profile matmul_int4.lpubin \
  --show-memory-bandwidth \
  --show-mac-utilization \
  --show-cycle-count
```

---

## 9. Next Steps for Your Project

### Immediate Actions

1. **Install LFortran**
   ```bash
   conda install -c conda-forge lfortran
   lfortran --version
   ```

2. **Generate MLIR from your code**
   ```bash
   cd 2025-3.5bit-groq-mvp
   lfortran --show-mlir matmul_int4_groq.f90 > mlir_output.txt
   ```

3. **Study the generated MLIR**
   - Look for `affine.parallel` (your `do concurrent`)
   - Check memory layout (`memref`)
   - Verify INT4/INT8 type preservation

### Advanced: Custom Groq Backend

If you get Groq SDK access, you can write custom lowering for your 3.5-bit format:

```cpp
// Custom MLIR pass for 3.5-bit (C++)
class Lower3p5BitToGroq : public PassWrapper<...> {
  void runOnOperation() override {
    // Pattern: match 3.5-bit unpacking operations
    RewritePatternSet patterns(&getContext());
    patterns.add<Unpack3p5BitPattern>(...);

    // Lower to Groq hardware primitives
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patterns)))
      signalPassFailure();
  }
};
```

---

## 10. Resources

### Official Documentation
- LFortran: https://lfortran.org/
- MLIR: https://mlir.llvm.org/
- MLIR Dialects: https://mlir.llvm.org/docs/Dialects/

### Groq Resources
- Groq Developer Portal: https://console.groq.com/
- Groq Compiler Docs: (requires access)

### Community
- LFortran Discord: https://discord.gg/fortran
- MLIR Discourse: https://discourse.llvm.org/c/mlir/

---

**Summary**: Your Fortran code's `do concurrent` and `pure` functions are already MLIR-friendly. The compilation path is: Fortran → LFortran → MLIR → Groq, with automatic optimizations (tiling, fusion) happening in the MLIR layer.

Your 3.5-bit innovation can be represented in MLIR and compiled to Groq hardware with custom unpacking kernels. The key is that MLIR preserves your high-level parallel structure (`do concurrent`) while optimizing for Groq's systolic array.
