# Lean Theorem Proving: AlphaProof MCTS + 3.5-bit Quantization

> **Status (2026-10-22):** the Lean files in this directory contain unproven goals (`sorry`), so they are sketches, not evidence of any proved property. Machine-checked results are in `../lean-kernel/`. See `../docs/CLAIMS_AUDIT_2026-10.md`.


## Overview
Formal theorem proving in **Lean 4** combining AlphaZero-style Monte Carlo Tree Search (MCTS) with proofs about 3.5-bit quantization correctness. Mathematical verification of AI algorithms.

## What to Add Here
- **Lean Proofs**: Formal theorem statements and proofs
- **MCTS Formalization**: AlphaZero tree search verified in Lean
- **Quantization Theorems**: 3.5-bit correctness proofs
- **Mathlib Integration**: Proofs building on Lean's math library
- **Documentation**: Proof strategies and insights

## Example Structure
```
lean-alphaproof-mcts/
├── README.md (this file)
├── LeanAlphaProof/
│   ├── MCTS/
│   │   ├── TreeSearch.lean
│   │   ├── UCB.lean (Upper Confidence Bound)
│   │   └── PolicyValue.lean
│   ├── Quantization/
│   │   ├── ThreeFiveBit.lean
│   │   ├── RoundingError.lean
│   │   └── InferenceCorrectness.lean
│   ├── NeuralNet/
│   │   ├── Matmul.lean
│   │   └── Activation.lean
│   └── Integration/
│       └── AlphaProofMCTS.lean
├── docs/
│   ├── proof_overview.md
│   ├── mcts_formalization.md
│   └── quantization_bounds.md
├── lakefile.lean (Lean build config)
└── lean-toolchain
```

## Key Theorems
1. **MCTS Convergence**: Proof that tree search converges to optimal policy
2. **Quantization Bounds**: Error bounds for 3.5-bit quantization
3. **Inference Correctness**: End-to-end proof of inference pipeline
4. **Matmul Properties**: Verified matrix multiplication properties

## Lean 4 Features
- **Dependent Types**: Precise specification of algorithms
- **Tactics**: Interactive and automated proving
- **Mathlib**: Building on extensive math library
- **Executability**: Proofs that can be extracted to code

## Connection to AlphaProof
Inspired by DeepMind's AlphaProof (IMO 2024 breakthrough), applying similar MCTS + proof search to verify AI inference algorithms.

## Theorem Categories
- **Algorithm Verification**: MCTS, neural network ops
- **Numerical Analysis**: Quantization error bounds
- **Correctness**: End-to-end inference properties
- **Optimization**: Proof that optimizations preserve semantics

## Getting Started
1. Add Lean 4 proof files
2. Document key theorems and lemmas
3. Include proof strategies and tactics used
4. Connect to SPARK verification for complete story

## Tools Required
- Lean 4 (latest stable)
- Lake build tool
- VSCode with Lean extension

---
*Mathematical certainty for AI inference: From MCTS to 3.5-bit quantization*
