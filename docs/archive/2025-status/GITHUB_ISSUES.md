# GitHub Issues to Create

Copy these to create issues on: https://github.com/jimxzai/asicForTranAI/issues/new

---

## Issue #1: Integrate KV Cache into Attention Loop

**Title:** `[FEATURE] Integrate KV cache into attention computation`

**Labels:** `enhancement`, `performance`

**Description:**
Currently KV cache infrastructure exists (arrays, init, reset functions) but is not integrated into the attention computation loop.

**Implementation Plan:**
1. Modify `grouped_query_attention()` to use cached K,V from previous positions
2. Update attention score loop to iterate over `[1:cache_pos+seq_len]` instead of `[1:seq_len]`
3. Store current K,V in cache after computation
4. Increment `cache_pos` for next iteration

**Files Affected:**
- `transformer_layer.f90` (lines 312-324)

**Priority:** High - Critical for efficient autoregressive generation

**Acceptance Criteria:**
- [ ] KV cache used during generation
- [ ] Speedup measured and documented
- [ ] Tests pass with caching enabled

---

## Issue #2: Download and Convert LLaMA 70B AWQ Weights

**Title:** `[FEATURE] Weight loading: Download LLaMA 70B AWQ weights`

**Labels:** `enhancement`, `weights`

**Description:**
Need to download real LLaMA 70B weights quantized to 4-bit AWQ format.

**Implementation Plan:**
1. Create Python script using `huggingface_hub`
2. Download from `TheBloke/Llama-2-70B-AWQ`
3. Convert safetensors to Fortran binary format
4. Generate metadata file with tensor shapes

**Files to Create:**
- `scripts/download_weights.py`
- `scripts/convert_weights.py`

**Priority:** Critical - Blocks real inference

**Acceptance Criteria:**
- [ ] Weights downloaded (~140GB)
- [ ] Converted to Fortran-readable format
- [ ] Metadata file generated
- [ ] README with instructions

---

## Issue #3: Implement Safetensors Weight Loader

**Title:** `[FEATURE] Load weights from safetensors/binary format`

**Labels:** `enhancement`, `weights`

**Description:**
Implement weight loading to populate the 80-layer model with real LLaMA weights.

**Implementation Plan:**
1. Option A: Pure Fortran binary reader
2. Option B: Python preprocessing + Fortran binary read
3. Load weights into `TransformerLayer` arrays (wq, wk, wv, wo, w_gate, w_up, w_down)

**Files Affected:**
- `llama_model.f90` - add `load_weights()` subroutine
- New: `weight_loader.f90`

**Priority:** Critical - Blocks real inference

**Acceptance Criteria:**
- [ ] All 80 layers loaded with correct weights
- [ ] Weight shapes validated
- [ ] Memory usage reasonable
- [ ] Loading time < 5 minutes

---

## Issue #4: Add SentencePiece Tokenizer Integration

**Title:** `[FEATURE] Integrate SentencePiece tokenizer`

**Labels:** `enhancement`, `tokenizer`

**Description:**
Add tokenization to convert text ↔ token IDs for LLaMA model.

**Implementation Plan:**
1. Option A: Python wrapper (easy)
2. Option B: Call SentencePiece C++ library from Fortran
3. Download LLaMA tokenizer model (`tokenizer.model`)
4. Implement encode/decode functions

**Files to Create:**
- `scripts/tokenize.py` (Python approach)
- OR `tokenizer.f90` (Fortran approach)

**Priority:** High - Needed for text generation

**Acceptance Criteria:**
- [ ] Text → token IDs working
- [ ] Token IDs → text working
- [ ] Special tokens handled (BOS, EOS)
- [ ] Test with example prompts

---

## Issue #5: Implement Sampling and Generation Loop

**Title:** `[FEATURE] Add text generation with sampling strategies`

**Labels:** `enhancement`, `generation`

**Description:**
Implement autoregressive text generation with various sampling strategies.

**Implementation Plan:**
1. Greedy sampling (argmax)
2. Temperature scaling
3. Top-k sampling
4. Top-p (nucleus) sampling
5. Generation loop with KV cache

**Files to Create:**
- `generation.f90` - sampling functions
- `llama_generate.f90` - main generation program

**Priority:** High - Core functionality

**Acceptance Criteria:**
- [ ] Greedy generation works
- [ ] Temperature scaling works
- [ ] Top-k/top-p implemented
- [ ] Generate coherent text (with real weights)
- [ ] Configurable max length, temperature, etc.

---

## Issue #6: End-to-End Inference Testing

**Title:** `[TEST] End-to-end inference with real weights`

**Labels:** `testing`, `validation`

**Description:**
Test complete pipeline: text input → tokenization → 80-layer inference → sampling → text output

**Test Cases:**
1. Single token generation
2. Short sequence (10 tokens)
3. Long sequence (100+ tokens)
4. Different sampling strategies
5. Multiple prompts

**Priority:** Critical - Validation

**Acceptance Criteria:**
- [ ] Pipeline runs without errors
- [ ] Generated text is coherent
- [ ] Performance measured (tok/s)
- [ ] Memory usage acceptable
- [ ] Comparison with reference implementation

---

## Issue #7: Performance Benchmarking and Optimization

**Title:** `[PERFORMANCE] Benchmark and optimize inference speed`

**Labels:** `performance`, `optimization`

**Description:**
Measure performance and optimize toward 3100+ tok/s target on Groq ASIC.

**Tasks:**
1. Benchmark current CPU performance
2. Profile bottlenecks
3. Optimize critical paths
4. Prepare for ASIC deployment (MLIR generation)

**Files Affected:**
- All core files for optimization
- New: `benchmark_inference.f90`

**Priority:** Medium - Post-MVP

**Acceptance Criteria:**
- [ ] CPU baseline established
- [ ] Bottlenecks identified
- [ ] Optimizations applied
- [ ] ASIC deployment path validated

---

## Quick Create Script

Run this to create all issues at once (requires `gh` CLI):

```bash
# Install GitHub CLI if needed
brew install gh
gh auth login

# Create issues (run from repo root)
gh issue create --title "[FEATURE] Integrate KV cache into attention computation" \
  --body "See GITHUB_ISSUES.md #1" --label "enhancement,performance"

gh issue create --title "[FEATURE] Download LLaMA 70B AWQ weights" \
  --body "See GITHUB_ISSUES.md #2" --label "enhancement,weights"

gh issue create --title "[FEATURE] Load weights from safetensors format" \
  --body "See GITHUB_ISSUES.md #3" --label "enhancement,weights"

gh issue create --title "[FEATURE] Integrate SentencePiece tokenizer" \
  --body "See GITHUB_ISSUES.md #4" --label "enhancement,tokenizer"

gh issue create --title "[FEATURE] Add text generation with sampling" \
  --body "See GITHUB_ISSUES.md #5" --label "enhancement,generation"

gh issue create --title "[TEST] End-to-end inference testing" \
  --body "See GITHUB_ISSUES.md #6" --label "testing,validation"

gh issue create --title "[PERFORMANCE] Benchmark and optimize" \
  --body "See GITHUB_ISSUES.md #7" --label "performance,optimization"
```
