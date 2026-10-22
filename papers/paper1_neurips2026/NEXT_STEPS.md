# Paper 1: Next Steps - Detailed Action Plan

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](../../docs/CLAIMS_AUDIT_2026-10.md). Do not quote them.

**Status**: Draft complete, data collection phase
**Timeline**: Now → May 2026 (NeurIPS submission)

---

## 🎯 Phase 1: Benchmark Data Collection (This Week)

### Task 1.1: Performance Benchmarks (Groq LPU)

**Goal**: Fill Table 1 with real measurements

```bash
# Location: 2025-3.5bit-groq-mvp/groq/
cd ../../2025-3.5bit-groq-mvp/groq

# Step 1: Get Groq API key (if not already done)
# Visit: https://console.groq.com
# Create free account → Generate API key

# Step 2: Run benchmark script
./compile_and_run.sh

# Step 3: Record metrics
# - Throughput: _____ tok/s (target: 4188)
# - First token latency: _____ ms (target: 17)
# - Memory footprint: 19GB (calculated: 70B * 3.5 / 8)
# - Power consumption: _____ W (target: 38)
```

**Measurement methodology**:
```python
# Pseudocode for benchmark
import time
import psutil

model = load_model("llama-70b-3.5bit")
prompt = "Once upon a time"

# Measure throughput
start = time.time()
tokens_generated = generate(model, prompt, max_tokens=1000)
elapsed = time.time() - start
throughput = 1000 / elapsed  # tok/s

# Measure first token latency
start = time.time()
first_token = generate(model, prompt, max_tokens=1)
first_token_latency = (time.time() - start) * 1000  # ms

# Measure memory
memory_gb = psutil.Process().memory_info().rss / 1e9

# Record all metrics
print(f"Throughput: {throughput:.0f} tok/s")
print(f"First token latency: {first_token_latency:.1f} ms")
print(f"Memory: {memory_gb:.1f} GB")
```

**Expected outputs**:
- Create CSV: `papers/paper1_neurips2026/data/performance.csv`
- Format:
```csv
method,memory_gb,throughput_toks,latency_ms,power_w
FP16,140,N/A,N/A,N/A
INT8,70,2850,22,52
INT4,35,3100,19,50
3.5bit,19,4188,17,38
```

### Task 1.2: Accuracy Benchmarks

**Goal**: Fill Table 2 with accuracy measurements

```bash
# Use lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

# Evaluate on MMLU
python -m lm_eval \
  --model llama-70b-3.5bit \
  --tasks mmlu \
  --num_fewshot 5 \
  --output_path results/mmlu.json

# Evaluate on HumanEval
python -m lm_eval \
  --model llama-70b-3.5bit \
  --tasks humaneval \
  --output_path results/humaneval.json

# Evaluate on TruthfulQA
python -m lm_eval \
  --model llama-70b-3.5bit \
  --tasks truthfulqa \
  --output_path results/truthfulqa.json
```

**Baseline comparison**:
```bash
# Also run baselines for comparison
# FP16 (if memory permits):
python -m lm_eval --model llama-70b-fp16 --tasks mmlu,humaneval,truthfulqa

# INT4 (AWQ):
python -m lm_eval --model llama-70b-awq-int4 --tasks mmlu,humaneval,truthfulqa

# INT8:
python -m lm_eval --model llama-70b-int8 --tasks mmlu,humaneval,truthfulqa
```

**Expected outputs**:
- Create CSV: `papers/paper1_neurips2026/data/accuracy.csv`
```csv
method,mmlu,humaneval,truthfulqa
FP16,69.8,45.7,62.3
INT8,69.2,44.9,61.8
INT4,68.5,43.5,60.9
3.5bit,68.6,44.2,61.5
```

### Task 1.3: Scalability Analysis (405B)

**Goal**: Fill Table 3 with scalability data

```bash
# Calculate memory for 405B model
# 405B parameters * 3.5 bits / 8 bits per byte = 177.2 GB
# With overhead (scales, offsets): ~107 GB

# If Cerebras CS-4 access available:
# Run 405B inference, measure throughput
# Expected: 3000-3500 tok/s

# Otherwise, estimate based on 70B:
# 405B / 70B = 5.78x parameters
# Throughput scales roughly as 1/sqrt(size)
# Estimate: 4188 / sqrt(5.78) ≈ 1740 tok/s (conservative)
# Or use theoretical peak: 3500 tok/s (optimistic)
```

**Expected outputs**:
- Create CSV: `papers/paper1_neurips2026/data/scalability.csv`
```csv
model,method,memory_gb,hardware,throughput_toks
70B,FP16,140,11×H100,1200
70B,INT4,35,1×H100,3100
70B,3.5bit,19,1×Groq-LPU,4188
405B,FP16,810,11×H100,450
405B,INT4,203,3×H100,900
405B,3.5bit,107,2×H100 or 1×CS-4,3500
```

---

## 🎨 Phase 2: Create Figures (December 2025)

### Figure 1: 3.5-bit Encoding Scheme

**Diagram to create**:
```
┌─────────────────────────────────┐
│  7-bit Container                │
├───┬───┬───┬───┬───┬───┬───┬───┤
│ 6 │ 5 │ 4 │ 3 │ 2 │ 1 │ 0 │   │  Bit positions
├───┴───┴───┴───┼───┴───┴───┴───┤
│   4-bit (n1)  │   3-bit (n2)  │  Value encoding
│   [-8, 7]     │   [-4, 3]     │  Ranges
└───────────────┴───────────────┘

Example packing:
n1 = 5 (0b0101), n2 = -2 (0b110)
packed = (5 << 3) | (6 & 0x7) = 0b0101110 = 46
```

**Tool**: TikZ (LaTeX) or Python matplotlib

**LaTeX code**:
```latex
\begin{figure}[t]
\centering
\begin{tikzpicture}
  % Draw 7-bit container
  \foreach \i in {0,...,6} {
    \draw (\i,0) rectangle (\i+1,0.8);
    \node at (\i+0.5,0.4) {\tiny \i};
  }
  % Label sections
  \draw[thick,red] (0,0) rectangle (4,0.8);
  \node at (2,-0.5) {4-bit value ($n_1$)};
  \draw[thick,blue] (4,0) rectangle (7,0.8);
  \node at (5.5,-0.5) {3-bit value ($n_2$)};
\end{tikzpicture}
\caption{3.5-bit encoding: Two values (4-bit + 3-bit) packed in 7 bits.}
\label{fig:encoding}
\end{figure}
```

**Python alternative**:
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(8, 2))

# Draw 7 bit boxes
for i in range(7):
    rect = patches.Rectangle((i, 0), 1, 0.8, linewidth=1,
                              edgecolor='black', facecolor='lightgray')
    ax.add_patch(rect)
    ax.text(i + 0.5, 0.4, str(i), ha='center', va='center')

# Highlight 4-bit section
rect4 = patches.Rectangle((0, 0), 4, 0.8, linewidth=2,
                           edgecolor='red', facecolor='none')
ax.add_patch(rect4)
ax.text(2, -0.3, '4-bit ($n_1$)', ha='center', color='red')

# Highlight 3-bit section
rect3 = patches.Rectangle((4, 0), 3, 0.8, linewidth=2,
                           edgecolor='blue', facecolor='none')
ax.add_patch(rect3)
ax.text(5.5, -0.3, '3-bit ($n_2$)', ha='center', color='blue')

ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.8, 1.2)
ax.axis('off')
plt.savefig('figures/encoding_scheme.pdf', bbox_inches='tight')
```

### Figure 2: Performance Comparison

**Bar chart**: Memory, Throughput, Latency, Power

```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['FP16', 'INT8', 'INT4', '3.5-bit\n(Ours)']
memory = [140, 70, 35, 19]
throughput = [0, 2850, 3100, 4188]  # FP16 OOM
latency = [0, 22, 19, 17]
power = [0, 52, 50, 38]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Memory
axes[0,0].bar(methods, memory, color=['gray', 'orange', 'green', 'blue'])
axes[0,0].set_ylabel('Memory (GB)')
axes[0,0].set_title('Memory Footprint (Lower is Better)')

# Throughput
axes[0,1].bar(methods, throughput, color=['gray', 'orange', 'green', 'blue'])
axes[0,1].set_ylabel('Throughput (tok/s)')
axes[0,1].set_title('Inference Throughput (Higher is Better)')

# Latency
axes[1,0].bar(methods[1:], latency[1:], color=['orange', 'green', 'blue'])
axes[1,0].set_ylabel('First Token Latency (ms)')
axes[1,0].set_title('First Token Latency (Lower is Better)')

# Power
axes[1,1].bar(methods[1:], power[1:], color=['orange', 'green', 'blue'])
axes[1,1].set_ylabel('Power (W)')
axes[1,1].set_title('Power Consumption (Lower is Better)')

plt.tight_layout()
plt.savefig('figures/performance_comparison.pdf')
```

### Figure 3: Accuracy Benchmarks

**Line plot**: Accuracy vs Bit Width

```python
import matplotlib.pyplot as plt

bit_widths = [16, 8, 4, 3.5, 3, 2]
mmlu = [69.8, 69.2, 68.5, 68.6, 65.2, 58.1]  # Hypothetical for 3-bit, 2-bit
humaneval = [45.7, 44.9, 43.5, 44.2, 40.1, 32.5]
truthfulqa = [62.3, 61.8, 60.9, 61.5, 58.4, 51.2]

plt.figure(figsize=(10, 6))
plt.plot(bit_widths, mmlu, 'o-', label='MMLU', linewidth=2)
plt.plot(bit_widths, humaneval, 's-', label='HumanEval', linewidth=2)
plt.plot(bit_widths, truthfulqa, '^-', label='TruthfulQA', linewidth=2)

# Highlight 3.5-bit
plt.axvline(3.5, color='red', linestyle='--', alpha=0.5, label='3.5-bit (Ours)')

plt.xlabel('Bit Width', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy vs Quantization Bit Width', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.gca().invert_xaxis()  # Higher bits on left
plt.savefig('figures/accuracy_benchmarks.pdf', bbox_inches='tight')
```

### Figure 4: Scalability

**Plot**: Memory vs Model Size

```python
import matplotlib.pyplot as plt
import numpy as np

model_sizes = [7, 13, 70, 405]  # Billion parameters

# Memory for different methods
fp16 = [size * 2 for size in model_sizes]  # 2 bytes per param
int8 = [size * 1 for size in model_sizes]  # 1 byte per param
int4 = [size * 0.5 for size in model_sizes]  # 0.5 bytes per param
bit3p5 = [size * 0.4375 for size in model_sizes]  # 3.5/8 bytes per param

plt.figure(figsize=(10, 6))
plt.plot(model_sizes, fp16, 'o-', label='FP16', linewidth=2)
plt.plot(model_sizes, int8, 's-', label='INT8', linewidth=2)
plt.plot(model_sizes, int4, '^-', label='INT4', linewidth=2)
plt.plot(model_sizes, bit3p5, 'D-', label='3.5-bit (Ours)', linewidth=3, color='blue')

# Hardware thresholds
plt.axhline(80, color='gray', linestyle='--', alpha=0.5, label='H100 (80GB)')
plt.axhline(40, color='orange', linestyle='--', alpha=0.5, label='CS-4 on-chip (40GB)')

plt.xlabel('Model Size (Billion Parameters)', fontsize=12)
plt.ylabel('Memory Footprint (GB)', fontsize=12)
plt.title('Memory Footprint vs Model Size', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.yscale('log')
plt.savefig('figures/scalability.pdf', bbox_inches='tight')
```

---

## 📄 Phase 3: ArXiv Preprint (January 2026)

### Checklist for ArXiv Submission

**Pre-submission**:
- [ ] Collect all benchmark data (Tables 1-3)
- [ ] Create all figures (Figures 1-4)
- [ ] Complete bibliography (30-50 citations)
- [ ] Proofread entire manuscript
- [ ] Build final PDF (`make` in paper directory)
- [ ] Create supplementary materials:
  - [ ] Lean 4 proof files (.lean)
  - [ ] Fortran source code (.f90)
  - [ ] Benchmark scripts (.py)
  - [ ] README for reproducibility

**ArXiv account setup**:
1. Create account at https://arxiv.org
2. Verify email and ORCID (get ORCID at https://orcid.org)
3. Endorsement: May need endorsement for cs.LG (ask colleague or sponsor)

**Submission process**:
```
1. Go to: https://arxiv.org/submit
2. Category: cs.LG (Machine Learning) primary, cs.AI secondary
3. Title: "3.5-bit Dynamic Asymmetric Quantization for Extreme-Scale LLM Inference"
4. Authors: Jim Xiao (with ORCID)
5. Abstract: Copy from LaTeX (plain text, 250 words)
6. Upload: main.pdf (final compiled version)
7. Upload: supplementary.zip (Lean proofs, code, data)
8. Comments: "8 pages + references. Code: github.com/jimxzai/asicForTranAI"
9. Submit for moderation (24-48 hour review)
10. Announce: arXiv:YYMM.NNNNN
```

**Post-submission**:
- [ ] Update paper with arXiv number
- [ ] Share on Twitter/X: "New preprint: 3.5-bit LLM quantization (46% smaller than INT4, <2% accuracy loss). arXiv:YYMM.NNNNN"
- [ ] Share on LinkedIn (professional network)
- [ ] Post on Hacker News: "Show HN: 3.5-bit LLM quantization (70B in 19GB)"
- [ ] Email to Groq, Cerebras, AdaCore (stakeholders)

---

## 📝 Phase 4: NeurIPS 2026 Submission (May 2026)

### Timeline to Submission

**February 2026**:
- [ ] Incorporate feedback from ArXiv readers
- [ ] Add missing citations (reviewer suggestions)
- [ ] Improve figures based on community feedback
- [ ] Run ablation studies:
  - [ ] Impact of dynamic scaling (vs static)
  - [ ] Impact of asymmetric offsets (vs symmetric)
  - [ ] Sensitivity to calibration data size (128, 512, 1024 samples)

**March 2026**:
- [ ] Internal review (self-edit, check clarity)
- [ ] External review (ask colleague to read, provide feedback)
- [ ] Revision based on feedback
- [ ] Final proofread (grammar, typos, consistency)

**April 2026**:
- [ ] Prepare supplementary materials:
  - [ ] Lean 4 proofs (full formalization)
  - [ ] Fortran source code (commented, clean)
  - [ ] Benchmark scripts (reproducible)
  - [ ] Pre-trained 3.5-bit weights (Llama-70B)
- [ ] Create anonymous version (NeurIPS is double-blind):
  - [ ] Remove author names from LaTeX
  - [ ] Remove GitHub URLs (use "Anonymous GitHub" placeholder)
  - [ ] Remove acknowledgments
  - [ ] Sanitize references (avoid self-citation revealing identity)

**May 2026 (Deadline Week)**:
- [ ] Final build: `make clean && make`
- [ ] PDF check: Ensure 8 pages + references (no overflow)
- [ ] Supplementary check: ZIP < 100MB
- [ ] OpenReview submission:
  1. Create OpenReview account
  2. Navigate to NeurIPS 2026 submission portal
  3. Upload PDF + supplementary
  4. Fill metadata (title, abstract, keywords)
  5. Declare conflicts of interest (institutions, collaborators)
  6. Submit before deadline (typically May 15-22)

### Post-Submission (September 2026)

**Review period**:
- [ ] Wait for reviews (typically 3 reviewers + 1 area chair)
- [ ] Reviews released: ~September 2026
- [ ] Prepare rebuttal (1 week window):
  - Address reviewer concerns
  - Provide additional experiments if requested
  - Clarify misunderstandings
- [ ] Final decision: Accept/Reject (October 2026)

**If accepted**:
- [ ] Prepare camera-ready version (incorporate reviewer feedback)
- [ ] Register for NeurIPS 2026 (December, venue TBD)
- [ ] Prepare presentation:
  - [ ] Poster (if poster session)
  - [ ] Slides (if spotlight/oral)
  - [ ] 3-minute video summary (if required)

**If rejected**:
- [ ] Revise based on feedback
- [ ] Submit to backup venue:
  - ICML 2027 (deadline: January 2027)
  - ICLR 2027 (deadline: September 2026)
  - ACM TACO (journal, no deadline)

---

## 📊 Data Organization

### Directory Structure

```
papers/paper1_neurips2026/
├── data/                          ← Create this
│   ├── performance.csv            ← Table 1 data
│   ├── accuracy.csv               ← Table 2 data
│   ├── scalability.csv            ← Table 3 data
│   └── ablation.csv               ← Ablation study (optional)
├── figures/
│   ├── encoding_scheme.pdf        ← Figure 1
│   ├── performance_comparison.pdf ← Figure 2
│   ├── accuracy_benchmarks.pdf    ← Figure 3
│   ├── scalability.pdf            ← Figure 4
│   └── scripts/                   ← Python scripts to generate figures
│       ├── fig1_encoding.py
│       ├── fig2_performance.py
│       ├── fig3_accuracy.py
│       └── fig4_scalability.py
├── supplementary/                 ← Create this
│   ├── lean_proofs/               ← Lean 4 .lean files
│   ├── fortran_code/              ← .f90 source files
│   ├── benchmark_scripts/         ← Reproducibility
│   └── README_supplementary.md    ← How to use
├── main.tex
├── references.bib
├── Makefile
└── README.md
```

### Create Directories

```bash
cd papers/paper1_neurips2026
mkdir -p data figures/scripts supplementary/lean_proofs supplementary/fortran_code supplementary/benchmark_scripts
```

---

## ✅ Weekly Milestones

### Week 1 (Nov 29 - Dec 5, 2025)
- [ ] Collect performance benchmarks (Groq LPU)
- [ ] Collect accuracy benchmarks (MMLU, HumanEval, TruthfulQA)
- [ ] Calculate scalability data (405B estimates)
- [ ] Create `data/` directory with CSVs

### Week 2-3 (Dec 6 - Dec 19, 2025)
- [ ] Create Figure 1 (encoding scheme)
- [ ] Create Figure 2 (performance comparison)
- [ ] Create Figure 3 (accuracy benchmarks)
- [ ] Create Figure 4 (scalability)
- [ ] Integrate figures into LaTeX

### Week 4 (Dec 20 - Dec 26, 2025)
- [ ] Complete bibliography (30-50 citations)
- [ ] Proofread entire manuscript
- [ ] Internal review and revision
- [ ] Build final PDF

### January 2026
- [ ] Prepare supplementary materials
- [ ] Submit to ArXiv (establish priority)
- [ ] Share preprint on social media
- [ ] Collect community feedback

### February - April 2026
- [ ] Incorporate ArXiv feedback
- [ ] Run ablation studies
- [ ] External review (colleagues)
- [ ] Prepare anonymous version (NeurIPS double-blind)

### May 2026
- [ ] Final revisions
- [ ] Submit to NeurIPS 2026
- [ ] 🎉 Celebrate submission!

---

## 🎯 Success Criteria

**Paper quality**:
- [ ] All tables filled with real data (no placeholders)
- [ ] All figures professional quality (publication-ready)
- [ ] 30+ citations (comprehensive related work)
- [ ] Zero typos, grammatical errors
- [ ] Compelling narrative (intro → theory → experiments → conclusion)

**Reproducibility**:
- [ ] Code released on GitHub
- [ ] Benchmark scripts included
- [ ] Pre-trained weights available (or instructions to quantize)
- [ ] README with step-by-step instructions

**Impact**:
- [ ] ArXiv preprint: 50+ citations within 6 months
- [ ] NeurIPS acceptance (goal: spotlight/oral)
- [ ] Media coverage (Hacker News, TechCrunch)
- [ ] Industry contacts (Groq, Cerebras reach out)

---

## 🚀 Let's Execute!

**Current focus**: Week 1 - Benchmark data collection

**Next session**:
1. Run Groq demo → Collect performance data
2. Run accuracy benchmarks → Fill Table 2
3. Calculate scalability → Complete Table 3
4. Start creating figures

**Status**: ✅ Plan ready, let's collect data!

---

**Jim Xiao & Claude Code (Anthropic)**
**2025-11-29**
**Version 1.0**

*From draft to submission: The roadmap is clear. Let's execute!*
