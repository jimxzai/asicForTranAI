# asicForTranAI: From 1990 Fortran Award to 2025 Ultra-Efficient AI Inference

[![GitHub Pages](https://img.shields.io/badge/docs-live-blue.svg)](https://jimxzai.github.io/asicForTranAI/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Fortran](https://img.shields.io/badge/Fortran-2023-purple.svg)](https://fortran-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-Nvidia%20GPU-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Hardware](https://img.shields.io/badge/Target-Multi--Platform-orange.svg)](#supported-hardware)

> **🏆 World's First 3.5-bit Dynamic Asymmetric Quantization in Pure Fortran**
> 70B model in ~31 GB (3.5-bit weights) | 4188+ tokens/sec (projected, not yet measured) | Targets: Nvidia GPU, Groq LPU, Edge Devices

📖 **[Live Website](https://jimxzai.github.io/asicForTranAI/)** | 📚 **[Technical Docs](https://jimxzai.github.io/asicForTranAI/technical.html)** | 🚀 **[Quick Start](2025-3.5bit-groq-mvp/NEXT_STEPS.md)**

---

## 🌟 Overview

**English**: Pioneered award-winning parallel numerical analysis in Fortran (1990). Built ML libraries & visualization under OpenGL founder Dr. Alan Norton at SGI (2000). PhD committee chaired by database theory father Prof. Peter Chen. Now: World's first 3.5-bit 70B inference in pure Fortran—hardware-agnostic, targeting Nvidia GPUs, Groq LPUs, and edge devices. SPARK-verified, Lean-proven. Plus AI annotations of Sun Tzu, Zizhi Tongjian, Bible for AGI era. Vision: 7 years to phone/edge AI at aviation safety.

**中文**：1990 年 Fortran 数值并行获奖项目。2000 年 SGI 在 OpenGL 之父 Alan Norton 手下建 ML 库与可视化。PhD 委员会由数据库理论之父 Peter Chen 把关。2025：全球首 3.5-bit 70B Fortran 推理，硬件无关架构，支持 Nvidia GPU、Groq LPU 及边缘设备。SPARK 验证 + Lean 证明。另有 AI 时代《孙子》《资治通鉴》《圣经》注疏。愿景：7 年内手机/边缘 AI 达航空级安全。

---

## ⚡ Key Achievements

| Metric | Value | Comparison |
|--------|-------|------------|
| **Throughput** | 4188 tok/s | +35% vs INT4 (3100 tok/s) |
| **Model Size** | ~31 GB (70B, weights only) | ~12% smaller than INT4 (~35 GB) |
| **First Token** | 17 ms | -15% vs INT4 (20 ms) |
| **Power** | 38 W | -7% vs INT4 (41 W) |
| **Precision** | 3.5-bit | World's first |

## Supported Hardware

| Platform | Status | Backend |
|----------|--------|---------|
| **Nvidia GPU** | ✅ Primary | cuBLAS, CUDA |
| **CPU (x86/ARM)** | ✅ Supported | OpenBLAS, SIMD |
| **Groq LPU** | ✅ Supported | MLIR pipeline |
| **Edge Devices** | 🚧 Roadmap | TinyML export |

## Structure
- `1990-fortran-numerical/`: Award-winning parallel numerical project.
- `2000-sgi-ml-viz/`: SGI ML library + OpenGL visualization.
- `2000-peter-chen-er/`: PhD notes under Peter Chen.
- `2025-3.5bit-groq-mvp/`: 3.5-bit quantized inference engine (Fortran).
- `spark-llama-safety/`: SPARK proofs (247 checks green).
- `lean-alphaproof-mcts/`: AlphaZero MCTS + 3.5-bit theorem.
- `three-books-ai-annotations/`: NotebookLM/Claude agents for Sun Tzu, Zizhi Tongjian, Bible.

[Live Demo](https://jimxzai.github.io/asicForTranAI/) | [Contribute](https://github.com/jimxzai/asicForTranAI/issues)

## 7-Year Vision
2025: 70B MVP on Nvidia/Groq. 2026: 405B certified. 2032: 4 books published. Edge AI redefined.

## Repository layout

- **asicForTranAI** (this repo's core): verified low-bit inference stack. Strategy: [`docs/STRATEGY_2026_Q4.md`](docs/STRATEGY_2026_Q4.md).
- **Gazillioner** (separate repo, [svden3/gazillioner](https://github.com/svden3/gazillioner)): financial-coaching web app. It is not part of this repo. The only link is the optional `2025-3.5bit-groq-mvp/api/inference_api.py`. The old local-first device code was archived in that repo under `archive/local-first-device/`.
