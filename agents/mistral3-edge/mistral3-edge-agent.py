#!/usr/bin/env python3
"""
Mistral 3 Edge Agent - Local Inference for Three-Books Annotations
用法：python mistral3-edge-agent.py "今天读《孙子·始计篇》，我的心得是……"

Author: Jim Xiao
Date: 2025-12-03
Model: ministral-3:8b (primary) / deepseek-r1:8b (fallback)
Platform: Ollama (Edge Deployment)

特点 Features:
- 🚀 Edge/Local 推理（无需云端，隐私优先）
- 🧠 Ministral 3:8b（256K 上下文，多语言，视觉能力）
- 🔄 自动降级到 DeepSeek-R1（推理优化）
- 📊 性能对比测试（vs Llama-3.3-70B @ Groq）
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Repository paths
REPO_ROOT = Path(__file__).parent.parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "three-books-ai-annotations"
BENCHMARK_DIR = REPO_ROOT / "agents" / "mistral3-edge" / "benchmarks"

# Model priority list (按优先级排序)
MODEL_PRIORITY = [
    "ministral-3:8b",      # 优先：Mistral 3 Edge (需 Ollama 0.13.1+)
    "ministral-3:14b",     # 备选：更强推理（如果资源足够）
    "deepseek-r1:8b",      # 降级：当前可用的推理优化模型
    "gpt-oss:20b",         # 降级：通用模型
]


def check_ollama_installed() -> bool:
    """检查 Ollama 是否安装"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_available_models() -> list[str]:
    """获取当前可用的 Ollama 模型"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return []

        # 解析输出，提取模型名称
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:  # 只有标题行
            return []

        models = []
        for line in lines[1:]:  # 跳过标题
            parts = line.split()
            if parts:
                models.append(parts[0])  # 第一列是模型名

        return models

    except Exception as e:
        print(f"⚠️  获取模型列表失败: {e}")
        return []


def select_best_model() -> Optional[str]:
    """根据优先级和可用性选择最佳模型"""
    available = get_available_models()

    print(f"📋 当前可用模型: {', '.join(available) if available else '(无)'}")
    print()

    for model in MODEL_PRIORITY:
        if model in available:
            print(f"✅ 选择模型: {model}")
            return model

    # 如果优先列表中都没有，尝试拉取 Ministral 3
    print("⚠️  优先模型不可用，尝试拉取 ministral-3:8b...")
    print("💡 这需要 Ollama v0.13.1+，当前可能失败")
    print()

    try:
        result = subprocess.run(
            ["ollama", "pull", "ministral-3:8b"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )

        if result.returncode == 0:
            print("✅ ministral-3:8b 拉取成功！")
            return "ministral-3:8b"
        else:
            print(f"❌ 拉取失败: {result.stderr}")
            print()
            print("=" * 60)
            print("🔧 解决方案：升级 Ollama 到 v0.13.1+")
            print("=" * 60)
            print("1. 访问: https://github.com/ollama/ollama/releases")
            print("2. 下载最新 pre-release (v0.13.1+)")
            print("3. 安装后运行: ollama pull ministral-3:8b")
            print()
            print("🔄 当前将使用备选模型...")
            print()

            # 返回第一个可用的备选模型
            if available:
                fallback = available[0]
                print(f"✅ 使用备选: {fallback}")
                return fallback
            else:
                return None

    except subprocess.TimeoutExpired:
        print("❌ 拉取超时")
        return None
    except Exception as e:
        print(f"❌ 拉取失败: {e}")
        return None


def call_ollama(model: str, prompt: str, temperature: float = 0.7) -> Tuple[str, dict]:
    """调用 Ollama 本地推理"""

    print(f"🤖 调用模型: {model}")
    print(f"🌡️  温度: {temperature}")
    print(f"📏 提示词长度: {len(prompt)} 字符")
    print()

    start_time = datetime.now()

    try:
        # 使用 ollama run 命令（支持流式输出）
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # DeepSeek-R1 是推理模型，需要更长时间（10分钟）
        # Mistral 3 则更快（3-5分钟）
        timeout = 600 if "deepseek" in model.lower() else 300
        stdout, stderr = process.communicate(input=prompt, timeout=timeout)

        if process.returncode != 0:
            print(f"❌ 推理失败: {stderr}")
            return None, {}

        elapsed = (datetime.now() - start_time).total_seconds()

        # 计算性能指标
        output_tokens = len(stdout.split())  # 粗略估计
        tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0

        metadata = {
            "model": model,
            "elapsed_seconds": round(elapsed, 2),
            "output_tokens_estimated": output_tokens,
            "tokens_per_second": round(tokens_per_sec, 2),
            "timestamp": datetime.now().isoformat()
        }

        print(f"✅ 推理完成")
        print(f"⏱️  耗时: {elapsed:.2f}秒")
        print(f"⚡ 速度: ~{tokens_per_sec:.1f} tokens/秒")
        print()

        return stdout.strip(), metadata

    except subprocess.TimeoutExpired:
        process.kill()
        timeout_str = "10分钟" if "deepseek" in model.lower() else "5分钟"
        print(f"❌ 推理超时（{timeout_str}）")
        if "deepseek" in model.lower():
            print("💡 DeepSeek-R1 是推理模型（chain-of-thought），处理长文本较慢")
            print("   建议：等待 Mistral 3 可用，或缩短输出要求")
        return None, {}
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return None, {}


def generate_sun_tzu_annotation(model: str, user_note: str) -> Tuple[Optional[str], dict]:
    """生成《孙子兵法》AI时代注疏（边缘推理版）"""

    prompt = f"""你现在是「AI时代孙子兵法注疏大师」，任务是：

## 用户心得：
{user_note}

## 你的任务：

1. **判断章节**: 自动识别属于《孙子兵法》哪一篇（如：始计篇、作战篇、谋攻篇等）
2. **引用原文**: 提供相关原文（中英对照）
3. **经典注疏**: 引用曹操注、杜牧注、李筌注等
4. **深度解读**: 2000-3000字分析（中英双语）
5. **2025 AI战例对照**:
   - Groq vs Nvidia（LPU vs GPU，"兵贵神速"）
   - Mistral 3 vs Llama 3（边缘部署，"不战而屈人之兵"）
   - AGI安全博弈（OpenAI vs Anthropic，"知彼知己"）

## 输出格式（Markdown）：

# 《孙子兵法·[篇名]》- 2025 AI时代注疏

**日期**: {datetime.now().strftime('%Y年%m月%d日')}
**模型**: Mistral 3 Edge (边缘推理)

## 一、原文引用

【中文】
[原文]

【English】
[Translation]

## 二、历代注疏

- **曹操注**: [引用]
- **杜牧注**: [引用]
- **现代解**: [引用]

## 三、深度解读（2000字，中英双语）

### 战略层面分析

[你的深度分析...]

### 战术层面映射

[具体战术...]

## 四、2025 AI时代对照

### 案例1: Groq vs Nvidia - "兵贵神速"
- **古义**: [解释]
- **今映**: Groq LPU 推理速度（500 tokens/s）vs Nvidia GPU（50-100 tokens/s）
- **启示**: [...]

### 案例2: Mistral 3 边缘部署 - "不战而屈人之兵"
- **古义**: [解释]
- **今映**: Ministral 3（3-14B）在 Jetson 边缘设备推理，无需云端"交战"
- **启示**: [...]

### 案例3: AGI安全 - "知彼知己，百战不殆"
- **古义**: [解释]
- **今映**: [...]
- **启示**: [...]

## 五、启示与思考（300字）

[总结...]

---

**标签**: #孙子兵法 #AI时代 #边缘AI #Mistral3 #开源模型

请直接输出完整的Markdown内容（无需代码块）。"""

    result, metadata = call_ollama(model, prompt)

    if result:
        metadata["category"] = "sun-tzu"
        metadata["user_note_length"] = len(user_note)
        metadata["output_length"] = len(result)

    return result, metadata


def save_annotation(content: str, metadata: dict) -> Path:
    """保存注疏到文件"""

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    model_name = metadata.get("model", "unknown").replace(":", "-")
    filename = f"{timestamp}-mistral3-edge-{model_name}.md"
    filepath = ANNOTATIONS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n---\n\n")
        f.write("## 性能元数据\n\n")
        f.write("```json\n")
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
        f.write("\n```\n")

    return filepath


def save_benchmark(metadata: dict) -> Path:
    """保存性能基准数据"""

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filepath = BENCHMARK_DIR / f"{timestamp}-benchmarks.jsonl"

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False))
        f.write("\n")

    return filepath


def main():
    print("=" * 70)
    print("  Mistral 3 Edge Agent - 边缘AI推理系统")
    print("  《孙子兵法》2025 AI时代注疏 - 本地隐私优先")
    print("=" * 70)
    print()

    # 检查 Ollama
    if not check_ollama_installed():
        print("❌ Ollama 未安装")
        print("请访问: https://ollama.com/download")
        sys.exit(1)

    print("✅ Ollama 已安装")
    print()

    # 选择最佳模型
    model = select_best_model()

    if not model:
        print("❌ 无可用模型")
        print()
        print("请尝试：")
        print("1. 升级 Ollama: https://github.com/ollama/ollama/releases")
        print("2. 或拉取备选模型: ollama pull deepseek-r1:8b")
        sys.exit(1)

    print()
    print("=" * 70)
    print()

    # 获取用户输入
    if len(sys.argv) > 1:
        user_note = " ".join(sys.argv[1:])
    else:
        print("请输入今天的《孙子兵法》心得（300-800字）：")
        print()
        user_note = input("> ")

    if not user_note.strip():
        print("❌ 心得不能为空")
        sys.exit(1)

    print()
    print(f"📝 收到心得 ({len(user_note)} 字)")
    print()
    print("=" * 70)
    print()

    # 生成AI注疏
    annotation, metadata = generate_sun_tzu_annotation(model, user_note)

    if not annotation:
        print("❌ 注疏生成失败")
        sys.exit(1)

    # 保存结果
    annotation_path = save_annotation(annotation, metadata)
    benchmark_path = save_benchmark(metadata)

    print("=" * 70)
    print("✅ 完成！")
    print("=" * 70)
    print()
    print(f"📄 注疏文件: {annotation_path.relative_to(REPO_ROOT)}")
    print(f"📊 性能数据: {benchmark_path.relative_to(REPO_ROOT)}")
    print()
    print(f"⏱️  耗时: {metadata.get('elapsed_seconds', 0)}秒")
    print(f"⚡ 速度: {metadata.get('tokens_per_second', 0):.1f} tokens/秒")
    print(f"📏 输出: {metadata.get('output_length', 0)} 字符")
    print()

    # 预览
    print("=" * 70)
    print("📖 注疏预览（前600字）：")
    print("=" * 70)
    print()
    preview = annotation[:600]
    print(preview)
    if len(annotation) > 600:
        print("\n... (查看完整文件以阅读剩余内容)")
    print()

    print("=" * 70)
    print("🎯 对比基准：Llama-3.3-70B @ Groq 云端 vs Mistral 3 @ 边缘")
    print("=" * 70)
    print(f"- 边缘延迟: {metadata.get('elapsed_seconds', 0)}秒（本次）")
    print("- 边缘隐私: ✅ 100%本地，无数据上云")
    print("- 边缘成本: ✅ $0/推理（vs 云端 $0.50-2/M tokens）")
    print("- 模型规模: 8B params（vs 70B云端）")
    print()
    print("💡 结论：边缘 AI 适合隐私场景、离线推理、成本敏感应用")
    print()

    # Git 提交提示
    print("=" * 70)
    print("💾 下一步：提交到仓库？")
    print("=" * 70)
    print()
    print(f"git add {annotation_path.relative_to(REPO_ROOT)}")
    print(f"git add {benchmark_path.relative_to(REPO_ROOT)}")
    print(f'git commit -m "feat: Add Mistral 3 edge annotation ({datetime.now().strftime('%Y-%m-%d')})"')
    print("git push origin main")
    print()
    print("🚀 7年传承计划 - 边缘AI赋能")
    print()


if __name__ == "__main__":
    main()
