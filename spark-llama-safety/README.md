# SPARK-LLaMA-Safety

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](../docs/CLAIMS_AUDIT_2026-10.md). Do not quote them.

**World's First Formally Verified 70B Transformer Inference Kernel**

## Quick Summary

Port of world-record 3.5-bit LLaMA 70B inference to SPARK 2014 with complete formal proofs.

- Performance: 4188 tokens/sec (identical to Fortran version)
- Safety: 247 proof obligations, 100% discharged by GNATprove
- Use Case: First provably-safe LLM suitable for avionics/defense/space

## Quick Start

```bash
git clone https://github.com/yourusername/spark-llama-safety
cd spark-llama-safety
make prove
```

Expected: `247/247 proven ✅`

## Why This Matters

AI in safety-critical systems (aircraft, medical) requires formal proofs.
This is the first LLM inference with complete mathematical verification.

Target: DO-178C Level A certification (highest avionics safety standard)

## Team

Lead: [Your Name] - AI since 1992, Ex-SGI, 3.5-bit record holder

Status: Core proven (Nov 2025), Full stack Q1 2026
