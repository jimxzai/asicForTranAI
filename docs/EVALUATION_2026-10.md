# System Evaluation — asicForTranAI, 2026-10-22

Scores are 0–10 and are my judgment against the rubric and evidence stated for each row. They are meant to show where the work is, not to be a benchmark. "Before" is the repository as found today; "After" is after the changes made in this session. Everything marked *measured* can be re-run with `make -C kernel3p5 test bench`, `python3 studies/format_error_study.py`, or `./scripts/claims_audit.sh`.

## Scorecard

| # | Dimension | Before | After | Evidence |
|---|---|---|---|---|
| 1 | Kernel correctness | 3 | 7 | Legacy 3.5-bit kernel stored 4 bits/weight and its model driver is dummy values. New reference kernels: exhaustive pack/unpack, bit-exact matmul vs integer/float64 references, bounds-checked. Not yet SIMD, no accuracy evaluation. |
| 2 | Formal verification evidence | 1 | 5 | "247/247 proven" was hard-coded output; the Ada files never compiled; 37 Lean `sorry`. Now: GNATprove proves 40 checks on real SPARK code and `lean-kernel` proves the same bounds and the pair-code round-trip, both with negative controls and the tight int32 bound. Gap: proofs are about models of the Fortran, not the Fortran; layout, matmul structure, scales and accuracy are unproved. |
| 3 | Alignment with industry formats | 2 | 5 | Industry moved to block-scaled FP4 (MXFP4/NVFP4) and vendor QAT checkpoints. MXFP4 reference kernel added (exact); NVFP4, FP8 and checkpoint loading (safetensors/GGUF) are not implemented. |
| 4 | Performance and hardware coverage | 2 | 4 | Measured CPU numbers (below). Nothing measured on GPU, Groq, Cerebras or any accelerator; no comparison to llama.cpp or vendor kernels. |
| 5 | Compiler and portability story | 3 | 3 | Scripts reference Flang/MLIR but neither tool is installed and nothing was run. Pure-Fortran kernels with a simple C ABI are portable in principle. Unchanged. |
| 6 | Testing and CI | 3 | 6 | CI was stubs that echoed results. Now kernel tests are a required job; SPARK, Lean and the claims audit run best-effort. 18 legacy test files are not run in CI. |
| 7 | Reproducibility and claim integrity | 1 | 7 | 7 audit failures found; public pages corrected; 80+ files bannered; benchmark records hardware, compiler, flags and reports median/min. About 70 files still repeat old figures behind banners. |
| 8 | Security and supply chain | 5 | 5 | Secret scan of tracked files is clean (one example key in docs). No dependencies, no untrusted-input parsing yet. Gaps: root `LICENSE` missing though the README says MIT (a MIT file exists in `2025-3.5bit-groq-mvp/`); no SBOM; Llama license text tracked alongside weights policy. |
| 9 | Repository structure | 2 | 6 | 551 tracked files incl. 150 build artifacts and 46 root status docs → 406 files, artifacts untracked, status docs archived, README rewritten. Still contains unrelated personal projects and a large legacy prototype. |
| 10 | Market and strategy fit | 3 | 6 | Dec 2025 plan was stale (Groq partnership, "zero competitors"). New thesis: verification for low-precision inference; funded adjacent players found; SBIR open. No customer or pilot evidence yet. |
| | **Average** | **2.5** | **5.4** | |

Reaching 8/10 overall needs, in order: (a) close the model-to-Fortran gap (prove the byte layout and matmul structure, or generate the Fortran from the verified model) (dimension 2 → 7), (b) a measured result on one real accelerator or GPU plus a comparison to an existing library (4 → 7), (c) accuracy evaluation of MXFP4 and the 3.5-bit format on a real small model (1 and 3 → 8), (d) one paid pilot (10 → 8).

## Findings that changed the design

### 1. The repo's 3.5-bit format is not competitive at honest bit budgets
`studies/format_error_study.py` (synthetic weights, fixed seed, scale overhead counted; heavy-tailed = Student-t, 3 d.o.f.):

| Format | bits/weight | SQNR, Gaussian | SQNR, heavy-tailed |
|---|---|---|---|
| INT3 g128 | 3.125 | 11.2 dB | 7.3 dB |
| 3.5-bit pair g128 | 3.625 | 13.5 dB | 10.5 dB |
| 3.5-bit pair g32 | 4.000 | 12.2 dB | 10.7 dB |
| INT4 g128 | 4.125 | 18.6 dB | 13.1 dB |
| MXFP4 g32 | 4.250 | 18.8 dB | 17.2 dB |
| INT4 g32 | 4.500 | 20.3 dB | 16.9 dB |
| NVFP4 g16 | 4.500 | 20.4 dB | 21.0 dB |

At 4.0 bits the pair format is 6 dB worse than INT4 g128 at 4.125. The 4-bit/3-bit alternation makes odd positions clip under a shared scale. A per-parity scale might help but adds overhead, and block-scaled FP4 already wins on heavy tails. Caveats: synthetic data, an approximate E4M3 scale rounding, no model accuracy. Decision: **MXFP4 becomes the primary verified target; the 3.5-bit format stays as a research variant.**

### 2. Measured CPU performance (single thread, scalar, 4096×4096 matvec, Intel i9-9980HK, gfortran 16.2 `-O3 -march=native`, median of 15, stable to ±1% across four runs)

| Kernel | ms/matvec | Time vs INT4 | Weight bytes vs INT4 |
|---|---|---|---|
| Packed INT4 | 13.3 | 1.00 | 1.000 |
| Packed 3.5-bit | 15.9 | 1.20 | 0.875 |
| MXFP4 (exact, E8M0 scales) | 15.6 | 1.17 | 1.063 |

Optimizations this session, all bit-exact: branch-free sign extension (3.5-bit −23%, INT4 −30% in absolute time); `scale()` instead of generic `2.0**n` plus a 16-entry signed decode table for MXFP4 (−40%). A 128-entry lookup-table decoder for the 3.5-bit format was **slower** (1.7× INT4) and was removed. Decode-bound scalar code is expected to be limited; SIMD or hardware bit-extraction could change the ranking, and none of this applies to accelerators.

### 3. Language choice for the middle layer (your Rust question)

**Recommendation: do not add a Rust middle layer now; define a stable C ABI and add Rust only for the one part that needs it later.**

- **Premature**: there is no end-to-end model run yet (the `llama70b_3p5bit.f90` driver is a skeleton). A middle layer is orchestration around kernels that don't produce a verified model output. Build the checked kernel and the loader first.
- **Where Rust earns its place**: parsing untrusted input (safetensors/GGUF model files, tokenizer data) and a serving runtime. Memory-safety bugs there are the realistic attack surface, and Rust removes most of that class. It calls Fortran through `iso_c_binding` with no friction.
- **Where it does not**: the safety-critical control path of the thesis. Ada/SPARK has decades of use under DO-178C and its formal-methods supplement, and this project's value is proofs plus certification evidence. Ferrocene (qualified Rust) is TÜV-qualified for ISO 26262 and IEC 61508, and supports customer efforts toward DO-178C DAL C, but I found no independent confirmation of an aviation qualification ([Ferrous Systems release notes](https://ferrous-systems.com/blog/ferrocene-26-02-0/), [The Register](https://www.theregister.com/2025/12/04/rust_core_library_partly_polished/)). Rust verification tools such as Kani and Verus exist; I did not research their maturity.
- **Cost**: this repo already has Fortran, Ada/SPARK, Lean and Python. A fifth language costs a solo maintainer more than it buys. The Gazillioner archive shows what that looks like (Go + Rust + Fortran, 14 compile errors in one crate).
- **Decision rule**: add Rust if (a) the product ingests files from outside, or (b) a customer requires Rust, and keep the Fortran–Rust boundary to a small, versioned C header.

### 4. Industry direction vs this project

| Industry signal (2026) | Fit |
|---|---|
| FP4 block-scaled formats standard on new GPUs; vendors ship QAT checkpoints | Now covered for MXFP4 (reference); NVFP4 and loading vendor checkpoints are next |
| Fixed-weight silicon (Taalas, being acquired by AMD) | Strong fit for pre-tapeout numerical verification; needs proofs, not benchmarks |
| ED-324 / ARP6983 airborne ML assurance expected ~Q4 2026; EU AI Act high-risk delayed to 2027–28 | Evidence-package format should follow ED-324 once published |
| Memory prices elevated through 2027–28 | Favors fewer bits/weight, but accuracy per bit says FP4 variants beat the pair format |
| Funded Lean/verification startups verify math and code, not numerics | Niche remains open; needs a machine-checked result to credibly claim it |

## Next engineering steps (priority order)
1. ~~Install Lean and GNATprove and prove the pair round-trip and accumulator bounds~~ (done 2026-10-22). Next: prove the 16-weights-in-7-bytes layout (`pack_column`/`unpack_column`) and the MXFP4 scale handling, and decide how to tie the proofs to the Fortran source.
2. Add NVFP4 (E4M3 block scale) reference kernel and tests; extend the format study to a real small model's weights.
3. Add a SIMD (or `do concurrent`) variant of each kernel and re-measure; compare against llama.cpp on the same weights.
4. Safetensors/GGUF reader; decide Rust vs Fortran for it using the rule above.
5. Add the root `LICENSE` (the README says MIT; confirm and copy the existing file), an SBOM, and run the legacy tests in CI or delete them.
