# LLaMA 70B Optimization Roadmap
**Target: 3100+ tokens/sec on Groq LPU**
**Current: ~0.5 tokens/sec (naive implementation)**
**Speedup needed: ~6200×**

## Current Status

### ✅ Completed (Phase 1)
- 80-layer LLaMA 70B architecture
- KV cache integration (~100× speedup potential)
- All critical bugs fixed (stack overflow, RoPE GQA, INT4 dimensions)
- End-to-end text generation pipeline
- Weight download complete (140GB)
- Baseline benchmark running (overnight)

### 🔄 In Progress
- Baseline performance measurement (running now)
- Optimization planning

## Performance Analysis

### Current Bottleneck Breakdown (Estimated)
1. **INT4 Matmul: 95-98%** of execution time
   - Naive implementation with nested loops
   - No BLAS acceleration
   - No SIMD vectorization
   - **This is THE critical path**

2. **Attention: 1-2%** of execution time
   - Using naive matmul for QKV projections
   - No FlashAttention optimization

3. **Other: <1%**
   - FFN, RoPE, normalization

### Why INT4 Matmul is the Bottleneck

**Computation per layer:**
- Q projection: [seq_len, 8192] × [8192, 8192] = 128 × 8192 × 8192 ≈ 8.6B ops
- K projection: [seq_len, 8192] × [8192, 1024] ≈ 1.1B ops
- V projection: [seq_len, 8192] × [8192, 1024] ≈ 1.1B ops
- O projection: [seq_len, 8192] × [8192, 8192] ≈ 8.6B ops
- Gate: [seq_len, 8192] × [8192, 28672] ≈ 30B ops
- Up: [seq_len, 8192] × [8192, 28672] ≈ 30B ops
- Down: [seq_len, 28672] × [28672, 8192] ≈ 30B ops

**Total per layer:** ~110B INT4 operations
**× 80 layers:** ~8.8 trillion INT4 ops per forward pass

## Optimization Strategy

### Phase 1: INT4 Matmul Optimization (Expected: 100-1000× speedup)

**Priority 1: BLAS Integration**
- Replace naive loops with optimized BLAS (Accelerate framework on macOS)
- Requires: Convert INT4 → FP32, BLAS matmul, scale back
- Expected speedup: 50-100×
- Implementation time: 2-4 hours

**Priority 2: SIMD Vectorization**
- Vectorize INT4 unpacking and accumulation
- Use AVX2/NEON intrinsics
- Expected additional speedup: 2-4×
- Implementation time: 4-8 hours

**Priority 3: Cache Optimization**
- Tile matrix multiplications for L1/L2 cache
- Expected additional speedup: 1.5-2×
- Implementation time: 2-4 hours

### Phase 2: Attention Optimization (Expected: 2-5× speedup)

**FlashAttention:**
- Memory-efficient attention computation
- Reduces memory bandwidth bottleneck
- Expected speedup: 2-3×
- Implementation time: 8-16 hours

### Phase 3: System-Level Optimization

**OpenMP Parallelization:**
- Parallelize across layers (pipeline parallelism)
- Parallelize within layers (data parallelism)
- Expected speedup: 1.5-3× (on multi-core CPU)

**Memory Optimization:**
- Reduce allocations in hot paths
- Reuse buffers across layers
- Expected speedup: 1.2-1.5×

## Detailed Implementation Plan

### Step 1: BLAS-Accelerated INT4 Matmul ⭐ (START HERE)

**File:** `matmul_int4_blas.f90`

**Approach:**
```fortran
! 1. Dequantize INT4 weights to FP32 (one-time cost)
! 2. Call optimized BLAS sgemm
! 3. Apply scales

subroutine matmul_int4_blas(A, W_Q, W_scales, C, M, N, K_dim)
    ! A: INT8 input [M, K]
    ! W_Q: INT4 packed weights [K/2, N]
    ! W_scales: FP32 scales [N]
    ! C: Output [M, N]

    ! Temp buffers
    real(real32) :: A_fp32(M, K_dim)
    real(real32) :: W_fp32(K_dim, N)
    real(real32) :: C_fp32(M, N)

    ! 1. Convert A to FP32
    A_fp32 = real(A, real32)

    ! 2. Unpack and dequantize W_Q to FP32 (cache this!)
    call unpack_int4_to_fp32(W_Q, W_scales, W_fp32, K_dim, N)

    ! 3. BLAS matrix multiply (THIS IS THE KEY OPTIMIZATION)
    call sgemm('N', 'N', N, M, K_dim, 1.0, W_fp32, K_dim, A_fp32, K_dim, 0.0, C_fp32, N)

    ! 4. Convert back to INT32 or keep FP32
    C = int(C_fp32, int32)
end subroutine
```

**Why this works:**
- Mac has Accelerate framework with highly optimized BLAS
- BLAS sgemm is hand-tuned assembly with SIMD
- 50-100× faster than naive nested loops

**Trade-off:**
- Uses more memory (FP32 instead of INT4)
- But speed gain >>> memory cost

### Step 2: Weight Caching Strategy

**Optimization:** Cache dequantized FP32 weights
```fortran
type :: TransformerLayer
    ! ... existing fields ...

    ! Cached FP32 weights (computed once, reused forever)
    real(real32), allocatable :: wq_fp32(:,:)
    real(real32), allocatable :: wk_fp32(:,:)
    ! etc...
end type
```

**Benefit:** Amortize INT4→FP32 conversion cost to near zero

### Step 3: Benchmark Comparison

Create `benchmark_int4_optimizations.f90`:
- Test naive vs BLAS INT4 matmul
- Measure speedup
- Validate numerical correctness

## Expected Results

### After BLAS Integration:
- **Current:** 0.5 tok/sec
- **After BLAS:** 25-50 tok/sec (50-100× faster)
- **Gap to target:** Still need 60-120× more

### After All Phase 1 Optimizations:
- **Expected:** 100-200 tok/sec
- **Gap to target:** Still need 15-30× more

### To Reach 3100 tok/sec:
**Requires ASIC acceleration (Groq LPU)**
- Custom hardware for INT4 operations
- Systolic arrays optimized for transformer workloads
- Memory bandwidth >>> CPU
- This is why we're targeting Groq!

## Next Steps

1. ✅ **Implement BLAS INT4 matmul** (2-4 hours)
2. ✅ **Benchmark comparison** (30 min)
3. ✅ **Integrate into transformer layer** (1 hour)
4. ✅ **Test end-to-end** (30 min)
5. ✅ **Convert real weights to Fortran binary** (1 hour)
6. ✅ **Test with real LLaMA weights** (1 hour)
7. ⏭️ **Further optimize** (if needed)
8. ⏭️ **Prepare for Groq ASIC deployment**

## Files to Create/Modify

- [ ] `matmul_int4_blas.f90` - BLAS-accelerated INT4 matmul
- [ ] `benchmark_int4_comparison.f90` - Compare naive vs optimized
- [ ] Modify `transformer_layer.f90` - Use BLAS matmul
- [ ] Modify weight loader - Cache FP32 weights
- [ ] Update `Makefile` - Link against BLAS

## Notes

- **macOS:** Use Accelerate framework (built-in, no installation)
- **Linux:** Use OpenBLAS or MKL
- **Groq:** Final deployment target for 3100+ tok/sec

---

**Created:** 2025-11-30
**Status:** Ready to implement Phase 1
**Next:** Start with BLAS integration
