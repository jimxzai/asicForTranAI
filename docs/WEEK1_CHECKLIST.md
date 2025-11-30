# Week 1 Execution Checklist (Nov 29 - Dec 5, 2025)

**Goal**: Launch Groq/Cerebras partnerships + validate benchmarks

---

## Monday, November 29 (TODAY)

### ✅ Completed
- [x] Create 1-page overview (ONE_PAGE_OVERVIEW.md)
- [x] Draft Groq email (GROQ_EMAIL_DRAFT.md)
- [x] Draft Cerebras email (CEREBRAS_EMAIL_DRAFT.md)

### 🎯 To Do
- [ ] **Send Groq email** (30 min)
  - Email: developer-relations@groq.com
  - Attach: ONE_PAGE_OVERVIEW.md as PDF
  - LinkedIn: Research Denis Abts (VP Engineering) for warm intro

- [ ] **Research Groq contacts on LinkedIn** (30 min)
  - Denis Abts (VP Engineering)
  - Jonathan Ross (CEO) - optional
  - Any mutual connections?

- [ ] **Convert overview to PDF** (15 min)
  ```bash
  # Option 1: Pandoc
  pandoc docs/ONE_PAGE_OVERVIEW.md -o docs/ONE_PAGE_OVERVIEW.pdf \
    --pdf-engine=xelatex -V geometry:margin=1in

  # Option 2: Online converter
  # Upload to https://www.markdowntopdf.com/
  ```

---

## Tuesday, November 30

### 🎯 To Do
- [ ] **Send Cerebras email** (30 min)
  - Email: research@cerebras.net
  - Attach: ONE_PAGE_OVERVIEW.md as PDF
  - LinkedIn: Research Andrew Feldman (CEO) for warm intro

- [ ] **Start patent draft** (4 hours)
  - Focus: Patent 1 (Formal Verification of Quantized MoE)
  - Sections: Abstract, Background, Key Claims, Detailed Description
  - Reference: docs/IMMEDIATE_ACTION_PLAN.md for claim structure

- [ ] **Research patent attorneys** (1 hour)
  - Specialization: AI/ML patents
  - Budget: $10K for provisional filing
  - Goal: Schedule call for Thursday

---

## Wednesday, December 1

### 🎯 To Do
- [ ] **Install lm-evaluation-harness** (1 hour)
  ```bash
  cd /Users/jimxiao/ai/asicForTranAI
  git clone https://github.com/EleutherAI/lm-evaluation-harness
  cd lm-evaluation-harness
  pip install -e .
  ```

- [ ] **Prepare weights for evaluation** (2 hours)
  - Convert 3.5-bit weights to HuggingFace format
  - Verify loading with transformers library
  - Test small inference run

- [ ] **Start MMLU benchmark** (6 hours - runs in background)
  ```bash
  python -m lm_eval --model hf \
    --model_args pretrained=weights/llama-70b-3.5bit \
    --tasks mmlu --batch_size 1 --device cpu
  ```

- [ ] **Continue patent draft** (4 hours)
  - Complete key claims section
  - Start detailed description with figures

---

## Thursday, December 2

### 🎯 To Do
- [ ] **Check MMLU results** (30 min)
  - Target: >67.5 score (<2% loss vs 68.9 FP16)
  - If <67.5: Debug quantization parameters
  - If >67.5: Celebrate and document! ✅

- [ ] **Run HumanEval benchmark** (2 hours - background)
  ```bash
  python -m lm_eval --model hf \
    --model_args pretrained=weights/llama-70b-3.5bit \
    --tasks humaneval --batch_size 1
  ```

- [ ] **Continue patent draft** (4 hours)
  - Finalize detailed description
  - Add technical diagrams (quantization pipeline)
  - Review for completeness

- [ ] **Patent attorney call** (1 hour)
  - Review draft claims
  - Discuss provisional filing timeline
  - Confirm budget ($10K)

---

## Friday, December 3

### 🎯 To Do
- [ ] **Update Paper 1 Table 3 with real MMLU results** (2 hours)
  - File: papers/paper1_neurips2026/main.tex
  - Replace projected values with actual scores
  - Update caption with "Validated on M2 Max (Dec 2025)"

- [ ] **Run TruthfulQA benchmark** (2 hours - background)
  ```bash
  python -m lm_eval --model hf \
    --model_args pretrained=weights/llama-70b-3.5bit \
    --tasks truthfulqa_mc --batch_size 1
  ```

- [ ] **Weekly review** (1 hour)
  - Document results in WEEK1_RESULTS.md
  - Update QUICK_START.md with progress
  - Plan Week 2 priorities

- [ ] **Groq/Cerebras follow-up** (30 min)
  - If no response: Send gentle follow-up
  - If response: Schedule calls for Week 2

---

## Success Criteria (End of Week 1)

### 🟢 Green Flags (Keep Going)
- ✅ Groq email sent (Monday)
- ✅ Cerebras email sent (Tuesday)
- ✅ MMLU shows <2% accuracy loss (>67.5 score)
- ✅ Patent draft 90% complete
- ✅ At least one response from Groq/Cerebras

### 🔴 Red Flags (Pivot)
- ❌ MMLU shows >3% accuracy loss (<66.5 score)
- ❌ Weights conversion fails (can't load with HuggingFace)
- ❌ No response from Groq/Cerebras by Friday
- ❌ Patent attorney says claims too broad

---

## Time Budget

| Day | Planned Hours | Tasks |
|-----|---------------|-------|
| Monday | 2 hours | Email prep, LinkedIn research, PDF conversion |
| Tuesday | 5.5 hours | Cerebras email, patent draft, attorney research |
| Wednesday | 7 hours | lm-eval setup, MMLU run, patent draft |
| Thursday | 7.5 hours | Results check, HumanEval, patent draft, attorney call |
| Friday | 5.5 hours | Paper update, TruthfulQA, weekly review |
| **Total** | **27.5 hours** | Full-time commitment this week |

---

## Quick Reference

**Key Files**:
- `docs/ONE_PAGE_OVERVIEW.md` - Technical overview (send to Groq/Cerebras)
- `docs/GROQ_EMAIL_DRAFT.md` - Ready to copy/paste
- `docs/CEREBRAS_EMAIL_DRAFT.md` - Ready to copy/paste
- `docs/IMMEDIATE_ACTION_PLAN.md` - Full 90-day roadmap
- `papers/paper1_neurips2026/main.tex` - Paper to update with results

**External Links**:
- Groq: https://groq.com/contact/
- Cerebras: https://cerebras.net/contact/
- lm-eval: https://github.com/EleutherAI/lm-evaluation-harness
- NeurIPS 2026: https://neurips.cc/Conferences/2026 (submission opens ~May 15)

---

**Last Updated**: 2025-11-29
**Status**: Ready to execute! 🚀

**Next Action**: Send Groq email using GROQ_EMAIL_DRAFT.md
