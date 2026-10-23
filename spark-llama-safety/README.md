# spark-llama-safety

SPARK code for the reference kernels in `../kernel3p5/`, proved with GNATprove.

| Package | What GNATprove proves (`make prove`) |
|---|---|
| `src/pack3p5.ads/.adb` | Pair-code decode is in range; `Encode` then decode returns the inputs; `Dot` over up to 2,097,151 terms (int8 activations × 3.5-bit weights) cannot overflow a 32-bit `Integer` and satisfies `abs result <= 1024 * terms` |
| `src/mxfp4.ads/.adb` | An MXFP4 block of 32 terms sums to at most 49,152 in magnitude (`32 * 128 * 12`) |

Result on 2026-10-22 (GNATprove 16.1.0 from Alire, level 2, CVC5): **all 40 checks proved** (28 run-time, 4 loop invariants, 3 functional contracts, 5 termination). Negative controls: raising `Max_Terms` to 2,097,152 makes the overflow check unprovable; a wrong table entry is rejected. Install and run: see [`../docs/INSTALL_GNAT.md`](../docs/INSTALL_GNAT.md).

## Limits
- These proofs are about this SPARK code, which mirrors the Fortran. They are not a proof about the Fortran source.
- Not covered: the 16-weights-in-7-bytes byte layout, the matmul loop structure and tiling, MXFP4 scale handling, NVFP4, floating-point accuracy.
- The Alire GNATprove is built from an intermediate SPARK Pro commit, not a specific certified release.
- `legacy/` holds the 2025 templates (`transformer_layer_safe`, `hip_wrapper_safe`, `quantization`, ...). They never compiled and are excluded from the project. The "247 proof obligations" figure in earlier documents did not come from running GNATprove.
