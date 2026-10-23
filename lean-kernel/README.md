# lean-kernel

Machine-checked statements about the reference kernels in `../kernel3p5/`, in core Lean 4 (no Mathlib).
Build: `lake build` (toolchain pinned in `lean-toolchain`; install with `brew install elan-init`).

| File | Proved |
|---|---|
| `LeanKernel/Pack3p5.lean` | `decode_encode`, `encode_decode`: the 7-bit pair code (4-bit even, 3-bit odd weight) round-trips for all 128 codes; weight ranges are exactly [-8,7] and [-4,3] |
| `LeanKernel/Bounds.lean` | `dot_natAbs_le`: |Σ aᵢwᵢ| ≤ A·B·K; `int32_safe_3p5`: int32 accumulator cannot overflow for K ≤ 2,097,151 with |a| ≤ 128, |w| ≤ 8; `int32_tight_3p5`: K = 2,097,152 can; `mx_block_bound`: an MXFP4 block of 32 stays within 49,152 |

Axioms used (checked with `#print axioms`): only `propext`, `Classical.choice`, `Quot.sound`; no `sorryAx`.

## What this does and does not establish
These proofs are about **Lean models** of the algorithm (`BitVec` encoding, list dot product). They are not a proof about the Fortran source: the link between `pack3p5.f90` and these definitions is by inspection plus the Fortran tests (`make -C ../kernel3p5 test`), which check the same properties exhaustively or against references. Closing that gap needs a verified path from the Fortran (for example the SPARK code in `../spark-llama-safety/` once proved) or a translation checked against it. Not covered yet: the pack/unpack byte layout (16 weights in 7 bytes), the matmul loop structure, MXFP4 scale handling, NVFP4, and any accuracy statement.
