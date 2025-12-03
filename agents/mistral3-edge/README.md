# Mistral 3 Edge Agent - 边缘AI推理系统

**日期**: 2025-12-03
**状态**: ✅ Ready (需 Ollama v0.13.1+ 才能使用 Ministral 3)
**目标**: 将 Mistral 3 开源模型部署到边缘设备，实现本地《孙子兵法》AI注疏

---

## 🎯 核心优势：为什么选择 Mistral 3 边缘部署？

| 维度 | Mistral 3 Edge | Llama-3.3-70B @ Groq Cloud |
|------|----------------|----------------------------|
| **隐私** | ✅ 100%本地，零数据上云 | ❌ 数据需传输到云端 |
| **成本** | ✅ $0/推理（仅硬件） | ⚠️  $0.50-2/百万 tokens |
| **延迟** | ✅ 无网络往返（本地推理） | ⚠️  网络延迟 + 队列等待 |
| **离线能力** | ✅ 随时可用（无需网络） | ❌ 必须联网 |
| **模型规模** | 8B-14B params | 70B params |
| **推理速度** | 52-273 tokens/s @ Jetson Thor | 500+ tokens/s @ Groq LPU |
| **多语言** | ✅ 中英法阿等数十种 | ✅ 主要英文，中文可用 |
| **视觉能力** | ✅ 256K上下文 + 图像分析 | ❌ 纯文本 |

**结论**：边缘适合隐私敏感、离线、成本优先场景；云端适合超大规模、极速推理、团队协作。

---

## 📦 前置要求

### 1. 安装 Ollama（当前：v0.13.0 → 需升级到 v0.13.1+）

#### 当前版本检查
```bash
ollama --version
# 输出: ollama version is 0.13.0
```

#### 升级到 Ollama v0.13.1+（支持 Ministral 3）

**⚠️  重要**：Ministral 3 模型需要 Ollama v0.13.1 或更高版本。当前稳定版是 v0.13.0，你需要：

**选项 A: 等待官方稳定版发布（推荐）**
- 访问：https://ollama.com/download
- 下载最新稳定版（v0.13.1+ 正式版发布后）

**选项 B: 使用 Pre-Release 版本（激进）**
- 访问：https://github.com/ollama/ollama/releases
- 下载 `v0.13.1-rc` 或更高的 pre-release 版本
- macOS: 下载 `Ollama-darwin.zip`
- Linux: 使用 `curl -fsSL https://ollama.com/install.sh | sh`（开发版通道）

**升级后验证**：
```bash
ollama --version
# 应显示: ollama version is 0.13.1 (或更高)

# 拉取 Ministral 3
ollama pull ministral-3:8b
```

---

### 2. 备选方案：当前可用的边缘模型

如果暂时无法升级 Ollama，本 Agent 自动降级到以下模型：

```bash
# 已安装的备选模型（按优先级）
ollama list

# 当前可用：
# - deepseek-r1:8b      ✅ 推理优化（数学/逻辑强）
# - gpt-oss:20b         ✅ 通用型（更大参数）
# - gpt-oss:120b        ✅ 超大模型（需 40GB+ 内存）
```

**Agent 会自动选择最佳可用模型**，无需手动配置。

---

## 🚀 使用方法

### 基础用法

```bash
cd agents/mistral3-edge

# 方式1：命令行参数
python mistral3-edge-agent.py "今天读《孙子·始计篇》，我的心得是：'兵者，诡道也'在AI时代意味着算法的不可预测性..."

# 方式2：交互式输入
python mistral3-edge-agent.py
# 然后粘贴你的心得
```

### 示例输出

```
======================================================================
  Mistral 3 Edge Agent - 边缘AI推理系统
  《孙子兵法》2025 AI时代注疏 - 本地隐私优先
======================================================================

✅ Ollama 已安装

📋 当前可用模型: deepseek-r1:8b, gpt-oss:20b, gpt-oss:120b

⚠️  优先模型不可用，尝试拉取 ministral-3:8b...
💡 这需要 Ollama v0.13.1+，当前可能失败

❌ 拉取失败: 412: The model you are attempting to pull requires a newer version

======================================================================
🔧 解决方案：升级 Ollama 到 v0.13.1+
======================================================================
1. 访问: https://github.com/ollama/ollama/releases
2. 下载最新 pre-release (v0.13.1+)
3. 安装后运行: ollama pull ministral-3:8b

🔄 当前将使用备选模型...

✅ 使用备选: deepseek-r1:8b

======================================================================

📝 收到心得 (87 字)

======================================================================

🤖 调用模型: deepseek-r1:8b
🌡️  温度: 0.7
📏 提示词长度: 1843 字符

✅ 推理完成
⏱️  耗时: 45.23秒
⚡ 速度: ~42.5 tokens/秒

======================================================================
✅ 完成！
======================================================================

📄 注疏文件: three-books-ai-annotations/2025-12-03-1430-mistral3-edge-deepseek-r1-8b.md
📊 性能数据: agents/mistral3-edge/benchmarks/2025-12-03-benchmarks.jsonl

⏱️  耗时: 45.23秒
⚡ 速度: 42.5 tokens/秒
📏 输出: 3245 字符

======================================================================
📖 注疏预览（前600字）：
======================================================================

# 《孙子兵法·始计篇》- 2025 AI时代注疏

**日期**: 2025年12月03日
**模型**: DeepSeek-R1 Edge (边缘推理)

## 一、原文引用

【中文】
兵者，诡道也。故能而示之不能，用而示之不用...

... (查看完整文件以阅读剩余内容)

======================================================================
🎯 对比基准：Llama-3.3-70B @ Groq 云端 vs Mistral 3 @ 边缘
======================================================================
- 边缘延迟: 45.23秒（本次）
- 边缘隐私: ✅ 100%本地，无数据上云
- 边缘成本: ✅ $0/推理（vs 云端 $0.50-2/M tokens）
- 模型规模: 8B params（vs 70B云端）

💡 结论：边缘 AI 适合隐私场景、离线推理、成本敏感应用

======================================================================
💾 下一步：提交到仓库？
======================================================================

git add three-books-ai-annotations/2025-12-03-1430-mistral3-edge-deepseek-r1-8b.md
git add agents/mistral3-edge/benchmarks/2025-12-03-benchmarks.jsonl
git commit -m "feat: Add Mistral 3 edge annotation (2025-12-03)"
git push origin main

🚀 7年传承计划 - 边缘AI赋能
```

---

## 📊 性能基准与对比

### 测试配置
- **硬件**: MacBook Pro / RTX PC / Jetson Thor
- **模型**: Ministral 3:8b (一旦可用) vs DeepSeek-R1:8b (当前备选)
- **任务**: 《孙子兵法》2000字注疏生成

### 预期性能指标（Ministral 3:8b）

| 平台 | 推理速度 | 延迟 | 功耗 | 成本 |
|------|---------|------|------|------|
| **Jetson Thor (边缘)** | 52-273 tokens/s | 15-60秒 | 10-25W | $0 |
| **RTX 4090** | 150-300 tokens/s | 10-30秒 | 200-300W | $0 |
| **MacBook Pro M3** | 30-80 tokens/s | 30-90秒 | 15-40W | $0 |

**对比 Groq 云端（Llama-3.3-70B）**：
- 速度：500+ tokens/s（快 2-10x）
- 成本：$0.50-2/百万 tokens（边缘 $0）
- 隐私：数据上云（边缘 100%本地）
- 离线：必须联网（边缘随时可用）

---

## 🔬 对 asicForTranAI 项目的战略意义

### 1. 验证 3.5-bit 量化栈（2025-3.5bit-groq-mvp/）
```bash
# 集成 Mistral 3 到你的 Fortran 优化栈
cd 2025-3.5bit-groq-mvp/
# 测试 Ministral 3:8b 的 3.5-bit 量化效果
# 目标：手机功耗（<10W）跑 100B 模型
```

**Mistral 3 优势**：
- 已针对 NVFP4（4-bit）量化优化 → 直接测试你的 3.5-bit Fortran matmul
- 256K 上下文 → 验证你的 `do concurrent` 并行优化
- Apache 2.0 开源 → 可集成到你的商业栈

### 2. 三书注疏加速（three-books-ai-annotations/）
- **当前**：Llama-3.3-70B @ Groq（每天 1 篇，3000字）
- **升级**：Ministral 3:8b 边缘（每天 3-5 篇，隐私优先，离线可用）
- **成果**：7年书稿积累量 3x 提升，2028年出版更丰富

### 3. Lean 4 定理验证（lean-alphaproof-mcts/）
```lean
-- 验证 Mistral 3 MoE 激活精度界
theorem mistral3_activation_bound (input : Float) :
  |mistral3_moe_activate input| ≤ 3.5_bit_precision := by
  sorry  -- 你的 SPARK 栈自动证明
```

### 4. 2028 NeurIPS 投稿目标
**论文标题**：*"Verified 3.5-bit Inference on Edge ASICs: Fortran to Mistral 3 via SPARK/Lean"*
- **贡献1**：首个 Fortran → MLIR → Mistral 3 全栈验证
- **贡献2**：边缘 ASIC 上跑 100B 模型（Jetson Thor 实测）
- **贡献3**：SPARK/Lean 形式化证明零误差累积

---

## 🛠️ 进阶配置

### 自定义模型优先级
编辑 `mistral3-edge-agent.py`：

```python
MODEL_PRIORITY = [
    "ministral-3:14b",     # 优先：14B 更强推理（如果你有 32GB+ RAM）
    "ministral-3:8b",      # 备选1
    "deepseek-r1:8b",      # 备选2
    # 添加你自己训练的模型
    "my-custom-model:8b",
]
```

### 集成视觉输入（Ministral 3 支持图像）
```python
# TODO: 扩展 Agent 支持《孙子》插图注疏
# 示例：上传古代战争地图，AI 分析"地形利用"
```

### 批量测试（基准对比）
```bash
# 生成 10 篇注疏，对比 Mistral 3 vs DeepSeek-R1
for i in {1..10}; do
    python mistral3-edge-agent.py "今天读《孙子·${i}篇》..."
done

# 查看性能数据
cat agents/mistral3-edge/benchmarks/*.jsonl | jq -s 'group_by(.model) | map({model: .[0].model, avg_speed: (map(.tokens_per_second) | add / length)})'
```

---

## 📚 相关资源

- **Mistral 3 官方发布**: https://mistral.ai/news/mistral-3/
- **Ollama 文档**: https://github.com/ollama/ollama
- **Ministral 3 基准测试**: https://huggingface.co/mistralai/Ministral-3-8B
- **NVIDIA Edge AI**: https://developer.nvidia.com/blog/mistral-3-edge-deployment/
- **本项目 Vision**: [VISION_VERIFICATION_2025-12-03.md](../../docs/VISION_VERIFICATION_2025-12-03.md)

---

## 🐛 常见问题

### Q1: 为什么 `ollama pull ministral-3:8b` 失败？
**A**: 你需要 Ollama v0.13.1+。检查版本：`ollama --version`。如果是 v0.13.0，请升级（见上文）。

### Q2: DeepSeek-R1 vs Mistral 3，哪个更好？
**A**:
- **DeepSeek-R1**: 推理优化（数学/逻辑强），当前可用
- **Mistral 3**: 多语言、视觉能力、256K上下文，但需 v0.13.1+

本 Agent 会自动选最佳可用模型。

### Q3: 推理速度慢怎么办？
**A**:
1. 用更小模型（3B）：`ollama pull ministral-3:3b`
2. 启用 GPU 加速：确保 CUDA/Metal 配置正确
3. 降低温度（更确定性）：修改代码 `temperature=0.3`

### Q4: 能在 Jetson/树莓派上跑吗？
**A**: ✅ 可以！Ministral 3 专为边缘设备优化。需：
- Jetson Thor/Orin: ✅ 完美（52-273 tokens/s）
- Jetson Nano: ⚠️  仅支持 3B 模型
- 树莓派 5: ⚠️  CPU 推理慢（建议 Jetson）

---

## 🎯 下一步

1. **立即测试**（用当前备选模型）：
   ```bash
   python agents/mistral3-edge/mistral3-edge-agent.py "今天读《孙子·始计篇》..."
   ```

2. **升级 Ollama**（等 v0.13.1 稳定版）：
   - 关注 https://ollama.com/download
   - 或使用 pre-release（激进用户）

3. **集成到 CI/CD**：
   ```yaml
   # .github/workflows/daily-annotations.yml
   - name: Run Mistral 3 Edge Agent
     run: python agents/mistral3-edge/mistral3-edge-agent.py "${{ secrets.DAILY_NOTE }}"
   ```

4. **Benchmark 对比**：
   - 每周生成 3 篇注疏
   - 对比 Groq 云端 vs Mistral 3 边缘
   - 2026 年发表 "Edge AI for Classical Texts" 论文

---

**Author**: Jim Xiao
**License**: Apache 2.0（与 Mistral 3 一致）
**Contact**: 见仓库主 README

**7年传承，边缘AI赋能** 🚀
