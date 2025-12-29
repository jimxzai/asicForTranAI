# MVP Checkpoints: 3.5-bit Inference Engine
**Updated: 2025-12-28** (Post Nvidia-Groq deal)

---

## Strategic Pivot Summary

The Nvidia-Groq licensing deal validates inference optimization as critical. Our strategy:
- **Primary**: Nvidia GPU (cuBLAS) - production-ready now
- **Secondary**: CPU (OpenBLAS/SIMD) - portable fallback
- **Tertiary**: Groq LPU - if API remains available post-acquisition

**Core Asset**: 3.5-bit quantization algorithm (hardware-agnostic)

---

## Phase 1: Core MVP (Weeks 1-4)

### Checkpoint 1.1: Verify Current Build
**Goal**: Confirm all existing code compiles and runs
```bash
cd 2025-3.5bit-groq-mvp
make clean && make all
make test
make benchmark-simd
```

**Exit Criteria**:
- [ ] All tests pass
- [ ] SIMD benchmark shows 6.995x speedup
- [ ] No compiler warnings

---

### Checkpoint 1.2: GPU Backend Validation
**Goal**: Verify Nvidia GPU acceleration works

```bash
make benchmark-blas    # CPU BLAS baseline
make benchmark-gpu     # Nvidia cuBLAS (if CUDA available)
```

**Exit Criteria**:
- [ ] cuBLAS matmul runs without errors
- [ ] Throughput > 1000 tok/s on GPU
- [ ] Document actual GPU specs and performance

---

### Checkpoint 1.3: End-to-End Inference Test
**Goal**: Generate actual text from LLaMA weights

```bash
make test-model
# Or with specific weights:
./llama_generate --weights weights/llama-70b-awq-int4/ --prompt "Hello"
```

**Exit Criteria**:
- [ ] Model loads without OOM
- [ ] Generates coherent text
- [ ] Latency < 500ms first token

---

## Phase 2: Performance Optimization (Weeks 5-8)

### Checkpoint 2.1: Baseline Metrics
**Goal**: Establish performance baseline across platforms

| Platform | Target | Metric |
|----------|--------|--------|
| CPU (M1 Max) | 100 tok/s | measure |
| CPU (x86 + AVX) | 150 tok/s | measure |
| Nvidia GPU (2080 Ti) | 4000 tok/s | measure |
| Nvidia GPU (4090) | 8000 tok/s | measure |

**Exit Criteria**:
- [ ] Benchmark script produces reproducible results
- [ ] Results documented in `benchmarks/` directory
- [ ] Comparison table vs INT4/INT8 baselines

---

### Checkpoint 2.2: Memory Optimization
**Goal**: Fit 70B model in target memory constraints

| Target | Memory Limit | Model Size (3.5-bit) |
|--------|--------------|----------------------|
| 24GB GPU | 22GB usable | 19GB (fits) |
| 16GB GPU | 14GB usable | 19GB (needs offload) |
| 8GB GPU | 7GB usable | Layer streaming required |

**Exit Criteria**:
- [ ] 24GB GPU: Full model in VRAM
- [ ] 16GB GPU: CPU offload working
- [ ] Memory profiling documented

---

### Checkpoint 2.3: Batch Processing
**Goal**: Optimize for throughput with batching

**Exit Criteria**:
- [ ] Batch size 1: Latency optimized
- [ ] Batch size 8: Throughput optimized
- [ ] Batch size 16+: Saturation point identified
- [ ] Trade-off curve documented

---

## Phase 3: Verification & Safety (Weeks 9-12)

### Checkpoint 3.1: Lean Proofs Complete
**Goal**: All theorem `sorry` placeholders resolved

```bash
cd ../lean-verification
lake build
```

**Exit Criteria**:
- [ ] `quantization_error_bound` proven
- [ ] `no_int32_overflow` proven
- [ ] `dequant_distributes` proven
- [ ] Zero `sorry` remaining

---

### Checkpoint 3.2: SPARK Contracts (Foundation)
**Goal**: Basic Ada safety layer compiles

**Exit Criteria**:
- [ ] GNAT installed and working
- [ ] Fortran-Ada FFI bridge compiles
- [ ] Basic matmul wrapped with Pre/Post contracts
- [ ] SPARK prover runs without errors

---

### Checkpoint 3.3: Test Coverage
**Goal**: Comprehensive test suite

**Exit Criteria**:
- [ ] Unit tests for all matmul variants
- [ ] Integration tests for transformer layer
- [ ] End-to-end generation tests
- [ ] Numerical accuracy tests (vs FP32 reference)
- [ ] Edge case tests (empty input, max sequence length)

---

## Phase 4: Documentation & Release (Weeks 13-16)

### Checkpoint 4.1: Technical Documentation
**Goal**: Complete docs for users and contributors

**Exit Criteria**:
- [ ] README updated with multi-platform support
- [ ] Installation guide (macOS, Linux, Windows)
- [ ] API reference for Fortran modules
- [ ] Architecture diagram

---

### Checkpoint 4.2: Benchmarks Published
**Goal**: Public benchmark results

**Exit Criteria**:
- [ ] Blog post: "3.5-bit Quantization: 46% Smaller, 10% More Accurate"
- [ ] Comparison vs llama.cpp, vLLM, TensorRT-LLM
- [ ] Reproducible benchmark scripts in repo

---

### Checkpoint 4.3: Release v1.0
**Goal**: Tagged release on GitHub

**Exit Criteria**:
- [ ] Version tagged: `v1.0.0`
- [ ] Release notes written
- [ ] Pre-built binaries (optional)
- [ ] Citation info (BibTeX)

---

## Decision Gates

### Gate 1 (Week 4): Go/No-Go for GPU Focus
**Question**: Is GPU performance meeting targets?

| Outcome | Action |
|---------|--------|
| GPU > 3000 tok/s | Continue GPU-first strategy |
| GPU 1000-3000 tok/s | Optimize, extend timeline |
| GPU < 1000 tok/s | Pivot to CPU-first, investigate |

---

### Gate 2 (Week 8): Platform Priority Decision
**Question**: Which platforms to support for v1.0?

| Scenario | Platforms for v1.0 |
|----------|-------------------|
| All targets met | Nvidia GPU + CPU + Groq API |
| GPU only meets targets | Nvidia GPU + CPU |
| CPU only meets targets | CPU only (delay GPU) |

---

### Gate 3 (Week 12): Certification Path Decision
**Question**: Pursue aerospace certification?

| Signal | Decision |
|--------|----------|
| Aerospace interest (LOI) | Invest in DO-178C path |
| No aerospace interest | Focus on commercial/cloud |
| Both | Dual track (more resources needed) |

---

## Weekly Status Template

```markdown
## Week N Status

### Completed
- [ ] Checkpoint X.Y achieved
- [ ] Specific deliverable

### In Progress
- [ ] Next checkpoint work
- [ ] Blockers: (list any)

### Metrics
| Metric | Target | Actual |
|--------|--------|--------|
| CPU tok/s | 100 | ? |
| GPU tok/s | 4000 | ? |
| Tests passing | 100% | ? |

### Decisions Needed
- (list any decisions for review)

### Next Week Plan
- (specific tasks)
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Groq API discontinued | Medium | Medium | Already have GPU backend |
| cuBLAS INT4 issues | Low | High | Have SIMD fallback |
| Memory overflow on 16GB GPU | Medium | Medium | Implement layer streaming |
| Lean proof stuck | Low | Low | Can ship without full proofs |
| SPARK integration complex | Medium | Medium | Start with minimal contracts |

---

## Quick Reference: Key Commands

```bash
# Build everything
make clean && make all

# Run tests
make test

# Benchmark CPU
make benchmark-simd

# Benchmark GPU (requires CUDA)
make benchmark-gpu

# Benchmark BLAS
make benchmark-blas

# Check Lean proofs
cd ../lean-verification && lake build

# Generate weights (if needed)
make gen-weights
```

---

---

## Publishing Schedule

### 2025-2026: Foundation Papers

| Paper | Venue | Deadline | Status |
|-------|-------|----------|--------|
| **ArXiv Preprint** | ArXiv | Jan 2025 | 🎯 Priority |
| **Paper 1: Theory** | NeurIPS 2026 | May 2026 | Draft |
| **Paper 2: Implementation** | ACM TACO | Mar 2026 | Outline |

### Publication Checkpoints

#### Q1 2025 (Jan-Mar)
- [ ] **CP-P1**: ArXiv preprint submitted (establish priority)
- [ ] **CP-P2**: Paper 1 draft complete (internal review)
- [ ] **CP-P3**: Benchmark data collected for all platforms

#### Q2 2025 (Apr-Jun)
- [ ] **CP-P4**: Paper 1 polished, co-author review
- [ ] **CP-P5**: Paper 2 draft started
- [ ] **CP-P6**: Blog post: "3.5-bit Quantization Explained"

#### Q1 2026 (Jan-Mar)
- [ ] **CP-P7**: Submit Paper 2 to ACM TACO (Mar deadline)
- [ ] **CP-P8**: Paper 1 final prep for NeurIPS

#### Q2 2026 (Apr-Jun)
- [ ] **CP-P9**: Submit Paper 1 to NeurIPS 2026 (May deadline)
- [ ] **CP-P10**: Start Paper 3 (Verification - CAV 2027)

### 6-Paper Series (Full Timeline)

```
2025 ──────────────────────────────────────────────────
  Jan: ArXiv preprint (priority)
  Mar: Blog posts, HN exposure

2026 ──────────────────────────────────────────────────
  Mar: Paper 2 → ACM TACO (Implementation)
  May: Paper 1 → NeurIPS 2026 (Theory)
  Dec: NeurIPS acceptance, presentation

2027 ──────────────────────────────────────────────────
  Jan: Paper 3 → CAV 2027 (Verification)
  Jul: Paper 4 → JSS (Certification)
  Oct: Paper 5 → IEEE Aerospace (Application)

2028 ──────────────────────────────────────────────────
  May: Paper 6 → CACM (Retrospective)
  Dec: Book proposal to MIT Press
```

### Publication Budget

| Item | Cost |
|------|------|
| Open Access Fees (6 papers) | $9,000 |
| Conference Travel (2 events) | $5,000 |
| Editing/Proofreading | $3,000 |
| **Total** | **$17,000** |

---

## Immediate Actions (This Week)

1. **Verify build** (Checkpoint 1.1)
2. **Prepare ArXiv preprint** (establish priority before competitors)
3. **Run GPU benchmarks** (data for Paper 1)

---

**Next Action**: Start with Checkpoint 1.1 - verify current build status.
