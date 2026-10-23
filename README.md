# asicForTranAI

Fortran reference kernels, formal proofs and tooling for **low-precision LLM inference you can check**.

The goal is narrow: for quantized formats (a 3.5-bit pair format today; MXFP4/NVFP4 and INT4 next), provide kernels in Fortran together with machine-checked statements that they cannot overflow and stay within stated error bounds. Bit-width is a parameter, not the product. **Current focus and the 14-day test: [`docs/FOCUS_2026-10.md`](docs/FOCUS_2026-10.md).** Longer plan: [`docs/STRATEGY_2026_Q4.md`](docs/STRATEGY_2026_Q4.md).

> **Status, 2026-10-22.** An audit of earlier claims ([`docs/CLAIMS_AUDIT_2026-10.md`](docs/CLAIMS_AUDIT_2026-10.md)) found that several headline numbers were wrong or unmeasured. Nothing below is claimed beyond what `make` reproduces.

## What works and is checked

| Item | State | How to check |
|---|---|---|
| 3.5-bit weight packing: 16 weights in exactly 7 bytes | Tested: exhaustive round-trip, bit-exact matmul vs naive integer reference, runtime bounds checks on | `make -C kernel3p5 test` |
| MXFP4 (OCP microscaling) block format and matmul | Tested: decode table, exact match to a float64 reference, scale-byte range, 4.25 bits/weight | same test |
| Storage is 3.5 bits/weight | Measured from array sizes; 70.6B params ≈ **30.9 GB** of weights | same test |
| int32 accumulator cannot overflow for K ≤ 2,097,151 (3.5-bit); MXFP4 block sum ≤ 49,152 | **Proved** twice: in Lean (`lean-kernel/`, `lake build`) and in SPARK (`make -C spark-llama-safety prove`: 46 checks, CVC5). The bound is tight (K = 2,097,152 is rejected by the prover) | see commands in the row |
| 16 weights ↔ 7 bytes layout: pack then unpack returns the same weights (one group, end to end) | **Proved** in Lean (`group_roundtrip`, via `bv_decide`, see trust note in `lean-kernel/README.md`); SPARK proves the bytes↔word half | `cd lean-kernel && lake build` |
| 7-bit pair code round-trips (`decode ∘ encode = id`), weight ranges exact | **Proved** in Lean (all 128 codes) and in SPARK (`Encode` postcondition) | same |
| CPU timing vs a packed INT4 baseline | Measured, single thread, scalar kernels (see below) | `make -C kernel3p5 bench` |

Latest local result (Intel i9-9980HK, gfortran 16.2, `-O3 -march=native`, one thread, scalar, 4096×4096 matvec, median of 15): packed INT4 13.3 ms; 3.5-bit 15.9 ms (1.20×, 12.5% fewer bytes); MXFP4 15.6 ms (1.17×, 6% more bytes). This says nothing about GPUs, accelerators or model accuracy. On synthetic heavy-tailed weights MXFP4 has about 4 dB better SQNR than INT4 g128 at similar bits, and the 3.5-bit pair format is worse than INT4 at equal bits (`studies/format_error_study.py`). Full scorecard: [`docs/EVALUATION_2026-10.md`](docs/EVALUATION_2026-10.md).

## What is not established

- **No end-to-end model run.** `2025-3.5bit-groq-mvp/llama70b_3p5bit.f90` is a skeleton with dummy values; only the quantization and matmul kernels are real.
- **Throughput on accelerators**: the 4,188 tok/s figure was an estimate and is not measured.
- **Proofs cover models, not the Fortran.** The Lean and SPARK artifacts mirror `kernel3p5/pack3p5.f90` and `mxfp4.f90`; the link to the Fortran source is by inspection plus the Fortran tests. They do not cover the matmul loop structure, MXFP4 scale handling, NVFP4, or accuracy.
- **Legacy proofs**: the earlier "247/247 proven" figure was hard-coded output and the 2025 Ada files were templates that never compiled (kept unproved in `spark-llama-safety/legacy/`). The legacy Lean directories contain 37 `sorry`.
- **Tool provenance**: the Alire GNATprove build is from an intermediate SPARK Pro commit, not a specific certified release.
- **Accuracy**: no perplexity or task evaluation of 3.5-bit vs INT4 on a real model has been done.

Re-run the full audit any time with `./scripts/claims_audit.sh` (currently 0 failures: no old claim is left unmarked, and GNATprove and Lean run; this is not a statement that everything is verified).

## Quick start

```bash
brew install gcc            # or any gfortran >= 12
./demo.sh                   # tests, then CPU benchmark
```

## Layout

| Path | Purpose |
|---|---|
| `kernel3p5/` | Reference 3.5-bit and MXFP4 kernels, tests, benchmark (start here) |
| `studies/` | Reproducible format error study (numpy) |
| `spark-llama-safety/` | SPARK code with contracts, proved by GNATprove (`src/`); unproved 2025 templates in `legacy/` |
| `lean-kernel/` | Lean 4 (core, no Mathlib) proofs: pair-code round-trip and accumulator bounds |
| `lean-verification/` | Older Lean 4 / Mathlib error-bound sketches with unproven goals (work in progress) |
| `2025-3.5bit-groq-mvp/` | Earlier prototype: kernels, GPU experiments, paper draft. Contains unaudited claims; treat as historical |
| `lean-alphaproof-mcts/` | Experiments with search-based proof generation |
| `docs/` | Strategy, audit, architecture notes; `docs/archive/` holds 2025 status documents |
| `1990-fortran-numerical/`, `2000-*`, `three-books-ai-annotations/`, `books-ai-publishing/` | Personal history and unrelated side projects |

Gazillioner, the financial-coaching product, is a separate repository ([svden3/gazillioner](https://github.com/svden3/gazillioner)). The only link is the optional `2025-3.5bit-groq-mvp/api/inference_api.py`.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Any performance or proof claim in a pull request needs a command that reproduces it.
