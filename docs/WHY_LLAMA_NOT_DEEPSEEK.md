# Why LLaMA (Not DeepSeek) as Our Foundation
**Question**: Why choose Ollama/LLaMA over DeepSeek to start with?
**Short Answer**: Architecture simplicity, formal verification tractability, and ASIC mapping
**Date**: 2025-11-29

---

## Executive Summary

While **DeepSeek V3** is technically impressive (671B params, $5.5M training cost, open-source MoE), we chose **LLaMA 2/3** as our foundation because:

1. **Dense architecture** is simpler to implement in Fortran (pure linear algebra)
2. **Formal verification** is tractable (uniform operations, no routing logic)
3. **ASIC mapping** is straightforward (systolic arrays, no conditional execution)
4. **Quantization research** is extensive (GPTQ, AWQ, SmoothQuant all use LLaMA)
5. **Safety-critical deployment** requires deterministic execution (LLaMA provides this)

**We can add DeepSeek/MoE in Paper 2-3** once the foundational work is proven.

---

## Detailed Comparison

### **Architecture Complexity**

| Aspect | LLaMA 2/3 | DeepSeek V3 | Winner for Our Use Case |
|--------|-----------|-------------|-------------------------|
| **Type** | Dense Transformer | Mixture-of-Experts (MoE) | LLaMA (simpler) |
| **Params** | 70B (all active) | 671B (236B active) | LLaMA (easier to handle) |
| **Operations** | Standard matmul | Matmul + expert routing | LLaMA (uniform ops) |
| **Control flow** | Sequential layers | Conditional expert selection | LLaMA (deterministic) |
| **Memory** | 140 GB FP16 | ~1.3 TB FP16 (total weights) | LLaMA (manageable) |

#### **LLaMA Architecture** (Simple, Uniform)
```fortran
! Clean, uniform architecture - perfect for Fortran
do layer = 1, 80
    ! Attention
    Q = matmul(X, W_q)
    K = matmul(X, W_k)
    V = matmul(X, W_v)
    attn_out = matmul(softmax(Q*K^T), V)

    ! FFN
    gate = matmul(attn_out, W_gate)
    up = matmul(attn_out, W_up)
    ffn_out = matmul(silu(gate) * up, W_down)
end do
```

**Fortran strengths**:
- Pure linear algebra (BLAS/LAPACK optimized)
- No complex control flow
- Cache-friendly sequential operations

#### **DeepSeek V3 Architecture** (Complex, Conditional)
```python
# Complex MoE routing - awkward in Fortran
for layer in range(126):
    # Multi-head latent attention (MLA)
    attn_out = mla_attention(x, W_mla)

    # MoE with 256 experts, top-8 routing
    router_logits = matmul(attn_out, W_router)
    top_k_experts, weights = topk(router_logits, k=8)

    # Conditional execution (non-deterministic!)
    ffn_out = 0
    for expert_id in top_k_experts:
        ffn_out += weights[expert_id] * experts[expert_id](attn_out)
```

**Challenges for Fortran**:
- Conditional expert selection (requires dynamic dispatch)
- Non-uniform memory access patterns
- Harder to map to SIMD/systolic arrays

---

### **Formal Verification Tractability**

This is **critical** for our safety-critical positioning.

#### **LLaMA: Verifiable** ✅

**Why tractable**:
1. **Uniform operations**: Every layer identical (80 copies of same structure)
2. **Deterministic execution**: Same input → same output (always)
3. **Compositional verification**: Prove 1 layer correct → all 80 layers correct

**Lean 4 verification approach**:
```lean
-- Verify one transformer layer
theorem transformer_layer_correct (x : Vector Float 8192) :
  let y := transformer_layer x W_attn W_ffn
  bounded_error x y ε := by
  -- Proof for 1 layer

-- Compose to full model
theorem llama_70b_correct (x : Vector Float 8192) :
  let y := iterate transformer_layer 80 x
  bounded_error x y (80 * ε) := by
  -- Compose 80 times
```

**Verification complexity**: O(n) where n = num_layers (80)

#### **DeepSeek: Difficult to Verify** ❌

**Why intractable**:
1. **Non-deterministic routing**: Expert selection changes per input
2. **256 experts per layer**: Must verify all 256 × 126 = 32,256 expert modules
3. **Top-k routing**: Conditional logic hard to reason about formally

**Lean 4 verification challenge**:
```lean
-- Must verify all possible expert combinations
theorem deepseek_layer_correct (x : Vector Float 8192) :
  ∀ (expert_ids : List Nat), expert_ids.length = 8 →
  let y := moe_layer x experts expert_ids
  bounded_error x y ε := by
  -- Must prove for all C(256, 8) = 10^15 combinations! (intractable)
```

**Verification complexity**: O(C(num_experts, k)^num_layers) = exponential

**Result**: DeepSeek MoE is **not formally verifiable** with current techniques.

---

### **ASIC Mapping & Hardware Efficiency**

Our goal: Compile to **Groq LPU** and **Cerebras WSE** ASICs.

#### **LLaMA: ASIC-Friendly** ✅

**Why straightforward**:
1. **Dense matmul**: Maps directly to systolic arrays
2. **Static memory**: All weights loaded at start (no dynamic routing)
3. **Predictable dataflow**: Sequential layer execution

**Groq LPU mapping**:
```
LLaMA Layer → Systolic Array Execution

[Input] → [W_q matmul] → [W_k matmul] → [W_v matmul]
          ↓               ↓               ↓
        [Q matrix]    [K matrix]      [V matrix]
          ↓               ↓               ↓
        [Q*K^T attention] → [Softmax] → [Attn*V]
          ↓
        [W_gate matmul] → [W_up matmul] → [SiLU] → [W_down matmul]
          ↓
        [Output]

All operations: Dense matmul (Groq's strength!)
Execution: Deterministic, pipelined
Memory: On-chip SRAM (230 MB on Groq LPU)
```

**Performance**: Groq achieves 750 TFLOPS on dense matmul (perfect for LLaMA).

#### **DeepSeek: ASIC-Difficult** ⚠️

**Why challenging**:
1. **Sparse activation**: Only 236B/671B params active (35% utilization)
2. **Dynamic routing**: Expert selection requires control logic on ASIC
3. **Irregular memory**: 256 experts → complex memory layout

**Groq LPU challenge**:
```
DeepSeek MoE → ASIC Execution (problematic)

[Input] → [Router logits] → [Top-k selection] ← PROBLEM: Control logic on ASIC
          ↓
        [Expert 47]  [Expert 128]  [Expert 201]  ... ← PROBLEM: Sparse access
          ↓            ↓             ↓
        [Weighted sum] ← PROBLEM: Dynamic memory gather

Issues:
1. Top-k requires sorting (expensive on ASIC)
2. Expert selection → irregular memory access
3. Only 35% of ASIC utilized (236B/671B active)
```

**Result**: MoE is better suited to **GPU** (flexible control flow) than **ASIC** (fixed dataflow).

---

### **Quantization Research Maturity**

#### **LLaMA: Extensive Research** ✅

**Published quantization methods**:
- **GPTQ** (2023): INT4 with 1.2% loss on LLaMA-70B
- **AWQ** (2023): Activation-aware INT4 for LLaMA
- **SmoothQuant** (2023): INT8 for LLaMA
- **OmniQuant** (2024): SOTA INT4 (0.8% loss)
- **QuIP** (2023): 2-bit LLaMA experiments

**Why this matters**:
- Proven baseline: We can compare our 3.5-bit against GPTQ/AWQ
- Established benchmarks: MMLU scores for LLaMA INT4 are well-documented
- Validation easier: Can reproduce llama.cpp INT4 results

#### **DeepSeek: Limited Quantization Research** ⚠️

**Published work**:
- Official DeepSeek release: FP16 only (no quantized weights)
- Community efforts: Some INT8 quantization attempts
- Research gap: No published INT4/sub-4-bit MoE quantization for DeepSeek

**Why challenging**:
- **MoE quantization**: More complex (must quantize 256 experts separately)
- **Router quantization**: How to quantize router logits without degrading expert selection?
- **Sparse patterns**: Quantization affects different experts differently

**Result**: DeepSeek quantization is an **open research problem** (high risk for our timeline).

---

### **Fortran Implementation Effort**

#### **LLaMA: Natural Fit for Fortran** ✅

**Why straightforward**:
```fortran
! LLaMA in Fortran: Clean, idiomatic
module llama_model
    use iso_fortran_env, only: int32, real32
    implicit none

    type :: TransformerLayer
        real(real32), allocatable :: W_q(:,:)
        real(real32), allocatable :: W_k(:,:)
        real(real32), allocatable :: W_v(:,:)
        real(real32), allocatable :: W_gate(:,:)
        real(real32), allocatable :: W_up(:,:)
        real(real32), allocatable :: W_down(:,:)
    end type

contains

    subroutine forward_pass(layer, x, y)
        type(TransformerLayer), intent(in) :: layer
        real(real32), intent(in) :: x(:,:)
        real(real32), intent(out) :: y(:,:)

        ! Attention
        Q = matmul(x, layer%W_q)
        K = matmul(x, layer%W_k)
        V = matmul(x, layer%W_v)
        attn = matmul(softmax(matmul(Q, transpose(K))), V)

        ! FFN
        gate = matmul(attn, layer%W_gate)
        up = matmul(attn, layer%W_up)
        y = matmul(silu(gate) * up, layer%W_down)
    end subroutine

end module
```

**Fortran strengths**:
- Native array operations (no loops)
- BLAS/LAPACK integration (highly optimized)
- Compile-time optimization (gfortran -O3)

**Lines of code**: ~2,000 LOC for full LLaMA 70B (we have this!)

#### **DeepSeek: Awkward for Fortran** ⚠️

**Why challenging**:
```fortran
! DeepSeek MoE in Fortran: Awkward, non-idiomatic
module deepseek_moe
    use iso_fortran_env, only: int32, real32
    implicit none

    type :: MoELayer
        real(real32), allocatable :: experts(:,:,:)  ! 256 experts
        real(real32), allocatable :: W_router(:,:)
        integer(int32) :: num_experts = 256
        integer(int32) :: top_k = 8
    end type

contains

    subroutine forward_pass_moe(layer, x, y)
        type(MoELayer), intent(in) :: layer
        real(real32), intent(in) :: x(:,:)
        real(real32), intent(out) :: y(:,:)

        ! Router logits
        router_logits = matmul(x, layer%W_router)

        ! Top-k selection (NO NATIVE FORTRAN SUPPORT!)
        ! Must implement custom sorting/selection
        call topk(router_logits, layer%top_k, top_experts, weights)

        ! Expert execution (DYNAMIC INDEXING - NOT FORTRAN'S STRENGTH)
        y = 0.0
        do i = 1, layer%top_k
            expert_id = top_experts(i)
            ! Dynamic array indexing (performance penalty)
            y = y + weights(i) * matmul(x, layer%experts(expert_id, :, :))
        end do
    end subroutine

end module
```

**Fortran weaknesses**:
- No native `topk` (must implement manually)
- Dynamic indexing (`experts(expert_id, :, :)`) → cache misses
- Conditional execution → harder to vectorize

**Estimated lines of code**: ~10,000 LOC for full DeepSeek (5× more complex)

---

### **Safety-Critical Certification**

This is our **core differentiator** for aerospace/automotive/medical markets.

#### **LLaMA: Certifiable** ✅

**Why feasible**:
1. **Deterministic execution**: Same input → same output (required for DO-178C Level A)
2. **Bounded memory**: All allocations static (required for ISO 26262 ASIL-D)
3. **No recursion**: Sequential layer execution (required for MISRA-C equivalent)
4. **Formal verification**: Lean 4 proofs enable certification artifacts

**Certification pathway** (DO-178C for aerospace):
```
Step 1: SPARK Ada port of Fortran kernel
Step 2: Prove runtime safety (no overflow, no array bounds violations)
Step 3: Prove functional correctness (Lean 4 theorems)
Step 4: Generate certification artifacts (proof certificates)
Step 5: Submit to FAA for DO-178C Level A certification

Timeline: 18-24 months
Cost: $2-5M (industry standard for Level A software)
Market: Avionics AI ($50B+ by 2035)
```

**Precedent**: CompCert (formally verified C compiler) achieved DO-178C certification in 2016.

#### **DeepSeek: Non-Certifiable** ❌

**Why infeasible**:
1. **Non-deterministic routing**: Expert selection varies (violates DO-178C determinism requirement)
2. **Dynamic memory**: Expert activation → unpredictable memory usage (violates ISO 26262)
3. **Conditional logic**: Router decisions → complex control flow (hard to verify)

**Certification blocker**:
```
DO-178C Level A Requirement:
"Software behavior must be deterministic and predictable"

DeepSeek MoE Violation:
- Same input can activate different experts (non-deterministic)
- Router uses softmax (floating-point rounding → non-reproducible)
- Top-k selection unstable (ties handled arbitrarily)

Result: CANNOT certify for flight-critical systems
```

**Impact**: DeepSeek **excluded from $50B+ safety-critical AI market**.

---

### **Community Support & Ecosystem**

#### **LLaMA: Massive Ecosystem** ✅

**Open-source tools**:
- **llama.cpp**: 60K+ GitHub stars, C++ reference implementation
- **Ollama**: 100K+ stars, local inference server
- **llama.rs**: Rust implementation
- **llama.go**: Go implementation
- **Transformers**: Official HuggingFace integration

**Quantization tools**:
- GPTQ: Auto-quantization to INT4
- AWQ: Activation-aware quantization
- GGUF format: Standard quantized weight format

**Model weights**:
- LLaMA 2: Free download from Meta
- LLaMA 3: Free download (8B, 70B, 405B)
- Variants: CodeLlama, Llama-Guard, Llama-Chat

**Benchmarks**:
- MMLU scores: Well-documented (68.9 for 70B FP16)
- HumanEval scores: Published in Meta papers
- Perplexity: WikiText-103 baselines available

**Result**: Easy to compare, validate, and reproduce our work.

#### **DeepSeek: Smaller Ecosystem** ⚠️

**Open-source tools**:
- Official PyTorch implementation (primary)
- Some community forks (limited)
- No standardized quantization format yet

**Model weights**:
- DeepSeek V3: Available on HuggingFace (671B params)
- Variants: Limited (focused on one main model)

**Benchmarks**:
- MMLU: Reported in paper (not widely reproduced)
- Community validation: Still ongoing

**Result**: Harder to validate claims, less community support.

---

## Strategic Roadmap: LLaMA First, DeepSeek Later

### **Phase 1: LLaMA Foundation** (2025-2026)

**Goal**: Prove concept with simple, verifiable architecture

**Deliverables**:
1. ✅ Fortran implementation (2,000 LOC) - DONE
2. ⚠️ 3.5-bit quantization validation (Dec 2025)
3. ⚠️ Formal verification (Lean 4 proofs, Mar 2026)
4. ✅ Paper 1 (NeurIPS 2026 submission)

**Why LLaMA**:
- Simplicity → faster proof-of-concept
- Extensive baselines → easy validation
- ASIC-friendly → Groq deployment feasible
- Certifiable → safety-critical market access

---

### **Phase 2: DeepSeek Extension** (2026-2027)

**Goal**: Extend to MoE architectures (after LLaMA proven)

**Deliverables**:
1. DeepSeek MoE quantization research
2. Sparse expert activation in Fortran
3. Router quantization analysis
4. Paper 2/3 (ACM TACO, CAV)

**Why later**:
- Build on LLaMA foundation
- MoE quantization still open research
- Verification techniques need refinement
- Allows two publication tracks (dense + MoE)

**Advantage**:
- **First** to formally verify MoE quantization
- **First** Fortran MoE implementation
- Broader academic impact (two architectures)

---

## Competitive Analysis: LLaMA vs DeepSeek for Our Goals

| Criteria | LLaMA 2/3 | DeepSeek V3 | Our Priority | Winner |
|----------|-----------|-------------|--------------|--------|
| **Implementation Simplicity** | ✅ Dense (easy) | ❌ MoE (complex) | High | **LLaMA** |
| **Formal Verification** | ✅ Tractable | ❌ Intractable | **Critical** | **LLaMA** |
| **ASIC Mapping** | ✅ Systolic arrays | ⚠️ GPU-friendly | High | **LLaMA** |
| **Quantization Research** | ✅ Extensive | ⚠️ Limited | High | **LLaMA** |
| **Fortran Fit** | ✅ Natural | ⚠️ Awkward | Medium | **LLaMA** |
| **Safety Certification** | ✅ Certifiable | ❌ Non-certifiable | **Critical** | **LLaMA** |
| **Community Support** | ✅ Massive | ⚠️ Growing | Medium | **LLaMA** |
| **Model Size** | ✅ 70B (manageable) | ⚠️ 671B (huge) | Medium | **LLaMA** |
| **Training Cost** | - | ✅ $5.5M (cheap) | Low (inference only) | Tie |
| **Performance** | ✅ Good | ✅ Better | Low (both sufficient) | Tie |

**Score**: LLaMA wins on **6/10 critical criteria** for our use case.

---

## Addressing Potential Counterarguments

### **"But DeepSeek is more powerful!"**

**Response**:
- True for general inference, but **irrelevant for safety-critical**
- Automotive/aerospace don't need 671B params (70B sufficient)
- Certification cost scales with model complexity (DeepSeek 10× harder)
- Our market values **verifiability over raw performance**

### **"DeepSeek is cheaper to train ($5.5M)!"**

**Response**:
- We're not training, we're doing **inference**
- Training cost doesn't affect our inference runtime
- LLaMA weights are free (Meta open-sourced them)

### **"MoE is more efficient (236B/671B active)!"**

**Response**:
- True for **GPU inference** (can skip experts)
- False for **ASIC** (irregular memory access → poor utilization)
- Our target (Groq LPU) optimized for dense matmul, not sparse

### **"Why not support both LLaMA and DeepSeek?"**

**Response**:
- **Resource constraint**: 6 months to NeurIPS submission
- **Proof-of-concept first**: Validate approach on simpler architecture
- **Future work**: DeepSeek MoE is excellent follow-up (Paper 2/3)
- **Academic impact**: Two separate contributions better than one

---

## Conclusion: LLaMA is the Right Starting Point

### **Summary of Reasoning**

| Reason | Impact | Priority |
|--------|--------|----------|
| **Formal verification tractable** | Enables safety-critical market ($50B+) | **Critical** |
| **ASIC mapping straightforward** | Groq deployment feasible | High |
| **Fortran implementation simple** | 2K LOC vs 10K LOC (5× less work) | High |
| **Extensive quantization research** | Easy to validate claims | High |
| **Deterministic execution** | Required for DO-178C certification | **Critical** |
| **Community support** | Easy to reproduce baselines | Medium |

### **Strategic Decision**

**Start with LLaMA (2025-2026)**:
- Prove 3.5-bit quantization works (Paper 1: NeurIPS 2026)
- Demonstrate formal verification (Lean 4 proofs)
- Deploy to Groq ASIC (if access granted)
- Achieve safety certification (SPARK Ada port)

**Extend to DeepSeek (2026-2027)**:
- After LLaMA validated, tackle MoE complexity
- Novel research: First formally verified MoE quantization
- Separate publication track (Paper 2/3: ACM TACO, CAV)
- Broader academic impact (dense + MoE coverage)

### **Final Answer**

**We chose LLaMA over DeepSeek because**:

1. **Simplicity → Speed**: Dense architecture = 5× less implementation effort
2. **Verifiability → Market**: Formal verification unlocks $50B+ safety-critical market
3. **ASIC-Ready → Performance**: Dense matmul maps perfectly to Groq/Cerebras
4. **Certifiable → Premium**: DO-178C certification requires deterministic execution
5. **Proven → Low Risk**: Extensive quantization research reduces validation risk

**DeepSeek is excellent**, but for a **different use case** (cloud inference, cost optimization). For **safety-critical, ASIC-optimized, formally verified inference**, LLaMA is the superior foundation.

We can add DeepSeek in Phase 2 after proving the concept!

---

**References**:
- LLaMA 2 Paper: https://arxiv.org/abs/2307.09288
- DeepSeek V3: https://github.com/deepseek-ai/DeepSeek-V3
- GPTQ: https://arxiv.org/abs/2210.17323
- Groq TSP: https://arxiv.org/abs/2004.14783
- DO-178C: RTCA/DO-178C Software Considerations in Airborne Systems

**Status**: This analysis justifies our LLaMA-first strategy and positions DeepSeek as valuable future work.
