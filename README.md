# asicForTranAI

Fortran reference kernels, formal proofs and tooling for **low-precision LLM inference you can check**.

The goal is narrow: for quantized formats (a 3.5-bit pair format today; MXFP4/NVFP4 and INT4 next), provide kernels in Fortran together with machine-checked statements that they cannot overflow and stay within stated error bounds. Bit-width is a parameter, not the product. See the plan in [`docs/STRATEGY_2026_Q4.md`](docs/STRATEGY_2026_Q4.md).

> **Status, 2026-10-22.** An audit of earlier claims ([`docs/CLAIMS_AUDIT_2026-10.md`](docs/CLAIMS_AUDIT_2026-10.md)) found that several headline numbers were wrong or unmeasured. Nothing below is claimed beyond what `make` reproduces.

## What works and is checked

| Item | State | How to check |
|---|---|---|
| 3.5-bit weight packing: 16 weights in exactly 7 bytes | Tested: exhaustive round-trip, bit-exact matmul vs naive integer reference, runtime bounds checks on | `make -C kernel3p5 test` |
| MXFP4 (OCP microscaling) block format and matmul | Tested: decode table, exact match to a float64 reference, scale-byte range, 4.25 bits/weight | same test |
| Storage is 3.5 bits/weight | Measured from array sizes; 70.6B params ≈ **30.9 GB** of weights | same test |
| int32 accumulator cannot overflow for K ≤ 2,097,151 | Tested boundary; not yet a machine-checked proof | same test; proof is the next milestone |
| CPU timing vs a packed INT4 baseline | Measured, single thread, scalar kernels (see below) | `make -C kernel3p5 bench` |

Latest local result (Intel i9-9980HK, gfortran 16.2, `-O3 -march=native`, one thread, scalar, 4096×4096 matvec, median of 15): packed INT4 13.3 ms; 3.5-bit 15.9 ms (1.20×, 12.5% fewer bytes); MXFP4 15.6 ms (1.17×, 6% more bytes). This says nothing about GPUs, accelerators or model accuracy. On synthetic heavy-tailed weights MXFP4 has about 4 dB better SQNR than INT4 g128 at similar bits, and the 3.5-bit pair format is worse than INT4 at equal bits (`studies/format_error_study.py`). Full scorecard: [`docs/EVALUATION_2026-10.md`](docs/EVALUATION_2026-10.md).

## What is not established

- **No end-to-end model run.** `2025-3.5bit-groq-mvp/llama70b_3p5bit.f90` is a skeleton with dummy values; only the quantization and matmul kernels are real.
- **Throughput on accelerators**: the 4,188 tok/s figure was an estimate and is not measured.
- **SPARK**: ~900 lines of Ada exist in `spark-llama-safety/`, but GNATprove has not been run here and the earlier "247/247 proven" output was hard-coded. CI now runs the real tool (see `.github/workflows/verify.yml`).
- **Lean**: the Lean sources contain 37 `sorry` (unproven goals), so they are not proofs yet.
- **Accuracy**: no perplexity or task evaluation of 3.5-bit vs INT4 on a real model has been done.

Re-run the full audit any time with `./scripts/claims_audit.sh`.

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
| `spark-llama-safety/` | Ada/SPARK contracts for the kernels (proofs pending) |
| `lean-verification/` | Lean 4 quantization error bounds (work in progress) |
| `2025-3.5bit-groq-mvp/` | Earlier prototype: kernels, GPU experiments, paper draft. Contains unaudited claims; treat as historical |
| `lean-alphaproof-mcts/` | Experiments with search-based proof generation |
| `docs/` | Strategy, audit, architecture notes; `docs/archive/` holds 2025 status documents |
| `1990-fortran-numerical/`, `2000-*`, `three-books-ai-annotations/`, `books-ai-publishing/` | Personal history and unrelated side projects |

Gazillioner, the financial-coaching product, is a separate repository ([svden3/gazillioner](https://github.com/svden3/gazillioner)). The only link is the optional `2025-3.5bit-groq-mvp/api/inference_api.py`.

## Contributing

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Any performance or proof claim in a pull request needs a command that reproduces it.
