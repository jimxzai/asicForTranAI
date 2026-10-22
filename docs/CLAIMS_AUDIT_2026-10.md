# Claims Audit — 2026-10-22

Reproduce with `./scripts/claims_audit.sh` (read-only; exits non-zero on failures). Result on this machine: 7 failures at first run, 6 after the fixes below (the rest need GNATprove, Lean and measurement).

**Bottom line**: the project's four headline claims are not supported by what the repo contains. Two are contradicted by arithmetic, one rests on hard-coded demo output, and one is partly unproven. The underlying engineering (a working 3.5-bit packing scheme, Fortran kernels, a Lean/SPARK skeleton) may be sound; the published numbers are not. This has to be fixed before any grant application, customer pilot or paper, because each of those involves checkable claims.

## Findings

| # | Claim (where repeated) | What the repo shows | Verdict |
|---|---|---|---|
| 1 | "70B in 19 GB, 46% smaller than INT4" | 70.6B params × 3.5 bits = **~30.9 GB**; INT4 is ~35.3 GB, so the saving is **~12%** at best, before scales and zero-points. 19 GB implies 2.15 bits/param. The repo's own `benchmark_report_3p5bit.md` table says **32.6 GB, 5.9% smaller than INT4**, and the same report's next table says 19 GB. | **Contradicted by arithmetic** |
| 2 | "4,188 tok/s on Groq, +28.9% vs INT4" | The benchmark report labels the Groq table **"Estimated"**. 4188/3124 is **+34.1%**, not 28.9%. The 28.9% is the speed-up from a small Python micro-benchmark (`benchmark_results_3p5bit.json`: 90 ms vs 70 ms), not a Groq measurement. `IMPLEMENTATION_COMPLETE.md` says "measured"; `COLD_EMAILS.md` says "simulated". No Groq run log is in the repo. | **Estimate presented as measurement; inconsistent** |
| 3 | "247 SPARK proofs, 100% discharged" | `spark-llama-safety/Makefile` `prove` and `demo` targets are `echo` statements, with the text "This is a placeholder" and "Full implementation coming Q1 2026". `llama_inference_template.ads` says "TODO: Run GNATprove to verify all 247 checks". `.github/workflows/verify.yml` SPARK steps are TODOs; CI never runs GNATprove. The Ada source is 7 files, 892 lines. `gnatprove` is not installed here, so nothing could be re-run. | **No evidence the number is real** |
| 4 | "Lean 4 proofs of quantization bounds" | 37 `sorry` (unproven goals) across `lean-verification` (6), `lean-alphaproof-mcts` (16) and `2025-3.5bit-groq-mvp/lean-proofs` (15), against 13 `theorem`/`lemma` declarations found. `lean`/`lake` not installed here, so the build was not run. | **Partly unproven** |
| 5 | "3.5-bit storage" (legacy kernel `matmul_3p5bit_dynamic.f90`) | `W_Q` is declared `integer(int8) :: W_Q(K/2, N)`: **one byte per pair of weights, i.e. 4 bits per weight**, one bit wasted per byte. As written it has no memory advantage over INT4. True 3.5-bit needs bit-packing across bytes (16 weights in 7 bytes). | **Not realised in the legacy kernel** |

## What was built in response (2026-10-22)

`kernel3p5/` is a reference implementation that makes the format real and checkable (`make -C kernel3p5 test`, `make -C kernel3p5 bench`):

- 16 weights in exactly 7 bytes; storage measured from array sizes at **3.500 bits/weight**, so 70.6B parameters = **30.9 GB** of weights.
- Exhaustive round-trip over all 8 positions × 128 seven-bit codes; random K=4096 round-trips; matmul **bit-exact** against a naive integer reference; runtime bounds checking on.
- int32 accumulator bound: safe for K ≤ 2,097,151 (|a| ≤ 128, |w| ≤ 8), and tight. Llama-70B has K ≤ 28,672. This is tested, not yet a machine-checked proof.
- CPU result (Intel i9-9980HK, gfortran 16.2, single thread, scalar, median of 15): the 3.5-bit matvec is **12.5% smaller** and **~20% slower** than packed INT4 on identical weights (4096×4096; ~29% slower before the sign-extension optimization). That contradicts "28.9% faster"; unpacking 7-bit fields costs more than splitting nibbles on a CPU. It says nothing about accelerators or accuracy. See `docs/EVALUATION_2026-10.md`.

## Where it is published (exposure)

- Public site and docs: `docs/index.html` (10 hits), `docs/technical.html` (5), `README.md` (4), `docs/blog_fortran_llm_2025.md` (6), `docs/EXECUTIVE_BRIEFING.md` (5), `docs/NEURIPS_2026_DRAFT_V1.md` (2), `paper/paper.tex` (1).
- Outreach: `COLD_EMAILS.md`, `docs/GROQ_EMAIL_DRAFT.md`, `docs/CEREBRAS_EMAIL_DRAFT.md`, `LINKEDIN_POST.md`. I can't tell from the repo which were actually sent or posted. If any were, consider a short correction to the recipients.
- About 70 files in total repeat at least one of the claims.

## Recommended order

1. **Today (no tools needed)**: stop sending or posting any of these figures. Remove or footnote them on `docs/index.html`, `docs/technical.html` and `README.md` (the pages the public sees). Suggested wording: "Design target: ~31 GB for 70B at 3.5 bits/weight; throughput figures are projections pending measurement."
2. **This week**: install GNAT/GNATprove (`spark-llama-safety/install_gnat.sh`) and Lean (`lake`); run `scripts/claims_audit.sh` again; record the real proof count and the real `sorry` list. Replace the Makefile `echo` demo with a real `gnatprove` call, and make CI run it and fail on unproven checks.
3. **Next**: one honest measurement: run the Fortran kernel on whatever hardware is available (local GPU or Groq cloud) and publish tok/s with the command, date, model and hardware.
4. Then update the paper draft and outreach from the measured numbers, and treat "verified" as meaning only what GNATprove and `lake build` report.

## Why this matters for the strategy

The Q4 2026 plan rests on "machine-checked guarantees". A buyer or reviewer will run the proof tools in the first hour. An honest 40-proof, 5-`sorry` result is a credible start. A claimed 247 that cannot be reproduced ends the conversation and can create legal exposure in grant or contract representations.
