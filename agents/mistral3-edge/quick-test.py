#!/usr/bin/env python3
"""
Quick Test - 快速验证 Mistral 3 Edge Agent 基础设施
用法：python quick-test.py

测试内容：
1. Ollama 安装检查
2. 可用模型检测
3. 简短推理测试（100字输出，<30秒）
"""

import subprocess
import sys
from pathlib import Path

def test_ollama():
    """测试1：Ollama 安装"""
    print("=" * 60)
    print("测试1：Ollama 安装检查")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ {version}")

            if "0.13.0" in version:
                print("⚠️  需要 v0.13.1+ 才能使用 Ministral 3")
                print("💡 当前使用备选模型测试")

            return True
        else:
            print("❌ Ollama 未正确安装")
            return False

    except FileNotFoundError:
        print("❌ Ollama 未安装")
        print("请访问: https://ollama.com/download")
        return False

def test_models():
    """测试2：可用模型"""
    print()
    print("=" * 60)
    print("测试2：可用模型检测")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print("❌ 无法获取模型列表")
            return []

        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            print("❌ 没有已安装的模型")
            return []

        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])

        print(f"✅ 找到 {len(models)} 个模型:")
        for model in models:
            marker = "🎯" if "ministral" in model.lower() else "📦"
            print(f"   {marker} {model}")

        return models

    except Exception as e:
        print(f"❌ 检测失败: {e}")
        return []

def test_quick_inference(models):
    """测试3：快速推理（100字输出）"""
    print()
    print("=" * 60)
    print("测试3：快速推理测试")
    print("=" * 60)

    if not models:
        print("❌ 无可用模型")
        return False

    # 选择第一个可用模型
    model = models[0]
    print(f"使用模型: {model}")
    print()

    # 简短提示词（只要100字输出）
    prompt = """请用100字简要解释《孙子兵法》中"知彼知己，百战不殆"在2025年AI领域的应用。
直接输出答案，无需额外格式。"""

    print("📝 提示词: 解释'知彼知己'在AI领域的应用（100字）")
    print("⏳ 预计 10-30 秒...")
    print()

    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 短超时（90秒），因为只要100字
        stdout, stderr = process.communicate(input=prompt, timeout=90)

        if process.returncode != 0:
            print(f"❌ 推理失败: {stderr}")
            return False

        print("✅ 推理成功！")
        print()
        print("=" * 60)
        print("输出预览：")
        print("=" * 60)
        print(stdout.strip()[:500])  # 只显示前500字符
        print()

        return True

    except subprocess.TimeoutExpired:
        process.kill()
        print("❌ 推理超时（90秒）")
        print("💡 这可能是推理模型（如 DeepSeek-R1），建议等待 Mistral 3")
        return False

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return False

def main():
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Mistral 3 Edge Agent - 快速测试                          ║")
    print("║  验证基础设施是否就绪                                      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # 测试1：Ollama
    if not test_ollama():
        print()
        print("❌ 测试失败：Ollama 未正确安装")
        sys.exit(1)

    # 测试2：模型
    models = test_models()

    # 测试3：推理
    success = test_quick_inference(models)

    # 总结
    print()
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    if success:
        print("✅ 所有测试通过！")
        print()
        print("🚀 下一步：")
        print("1. 运行完整 Agent:")
        print("   python agents/mistral3-edge/mistral3-edge-agent.py \"你的心得...\"")
        print()
        print("2. （推荐）升级 Ollama 到 v0.13.1+ 以使用 Ministral 3:")
        print("   https://github.com/ollama/ollama/releases")
        print()
        print("3. 或拉取备选模型（如果当前模型较慢）:")
        print("   ollama pull ministral-3:3b  # 更快（需 v0.13.1+）")
        print()
    else:
        print("⚠️  部分测试失败")
        print()
        print("可能原因：")
        print("- 当前模型是推理型（DeepSeek-R1），处理慢")
        print("- 需要升级 Ollama 到 v0.13.1+ 使用 Ministral 3")
        print()
        print("建议：")
        print("1. 等待 Ollama v0.13.1 正式版")
        print("2. 或使用预览版: https://github.com/ollama/ollama/releases")

    print()

if __name__ == "__main__":
    main()
