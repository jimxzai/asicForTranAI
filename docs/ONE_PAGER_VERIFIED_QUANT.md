# Verified arithmetic for quantized LLM inference

**Problem.** Low-bit inference (INT4, MXFP4, NVFP4, custom formats) moves correctness into small kernels: bit-packing, decode tables, and int32 or float accumulation. Bugs here are silent. Overflow and decode errors do not crash; they change outputs. On fixed-weight silicon a mistake is a respin.

**What we do.** For a given quantized format and dot-product kernel we deliver machine-checked proofs (SPARK/GNATprove and Lean 4) that the code cannot overflow, that decode inverts encode, and that accumulators stay within stated bounds, plus a bit-exact test oracle.

**What exists today** (reproducible; commands in the repository):
- Reference Fortran kernels for a 3.5-bit pair format and OCP MXFP4, tested bit-exact against integer / float64 references; exhaustive where the domain is finite.
- GNATprove proves 46 checks on SPARK models of those kernels: int32-overflow-freedom for up to 2,097,151 terms (the bound is tight: one more term is unprovable), the MXFP4 block bound of 49,152, and encode/decode round-trip. The same facts, plus the 16-weights-in-7-bytes packing layout, are proved independently in Lean 4 (the layout proof uses a SAT-based tactic whose certificate check is an added trusted component).
- A numerical study of weight-quantization error across formats at honest bit budgets, and CPU timings with medians.

**What it is not.**
- The proofs are about models that mirror the Fortran, not about the Fortran source itself; the link is by inspection and tests.
- No accuracy evaluation on a real model, no accelerator measurements, no NVFP4 yet.
- The GNATprove used is an Alire build from an intermediate SPARK Pro commit, not a certified release. For certification evidence the qualified product would be used.

**Engagement.** Fixed-scope review of one format and one kernel; deliverables are proofs, the oracle, and a short report stating exactly what is and is not covered. Scope and terms to be agreed after a first call.

**Why us.** Numerical-analysis and parallel-Fortran background, and an audit habit: the repository publishes the checks that falsify its own earlier claims.

Repository: https://github.com/jimxzai/asicForTranAI · Start with `README.md` and `./scripts/claims_audit.sh`.
