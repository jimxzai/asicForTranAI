#!/usr/bin/env python3
"""
Daily Three-Books Agent - Ministral-3 LOCAL (Ollama)
用法：python daily-annotation-ollama.py "今天读《孙子·始计篇》……"

本地运行，完全免费，无需API key，数据不出你的机器。

Author: Jim Xiao
Date: 2025-12-03
Model: ministral-3:8b (or any Ollama model)
Speed: 80-200 tok/s on RTX 4090, 30-80 tok/s on M2/M3 Mac
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Repository paths
REPO_ROOT = Path(__file__).parent.parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "three-books-ai-annotations"
DRAFTS_DIR = REPO_ROOT / "books-ai-publishing" / "drafts"

# Default model (can be changed)
DEFAULT_MODEL = "ministral-3:8b"  # Change to "llama3.3:70b" or others if you prefer


def check_ollama():
    """检查Ollama是否安装并运行"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """调用本地Ollama模型"""

    if not check_ollama():
        print("❌ Ollama未安装或未运行")
        print("安装: https://ollama.com/download")
        print("然后运行: ollama pull ministral-3:8b")
        sys.exit(1)

    print(f"🤖 正在调用本地模型: {model}")
    print("💻 完全本地运行，数据不出你的机器")
    print()

    try:
        # Use subprocess to call ollama
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=prompt, timeout=180)

        if process.returncode != 0:
            print(f"❌ Ollama错误: {stderr}")
            sys.exit(1)

        return stdout.strip()

    except subprocess.TimeoutExpired:
        print("❌ 生成超时（3分钟），请重试")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 调用Ollama失败: {e}")
        sys.exit(1)


def generate_annotation(user_note: str, model: str = DEFAULT_MODEL) -> tuple[str, dict]:
    """生成AI注疏"""

    prompt = f"""你现在是「AI时代三书注疏大师」，任务是：

1. **自动分类**: 判断用户心得属于《孙子兵法》《资治通鉴》《圣经》中的哪一部、哪一篇/卷/章
2. **检索原文**: 引用相关原文（中英对照）
3. **历代注疏**: 引用经典注疏（如曹操注、杜牧注、胡三省注、马丁·路德注释等）
4. **深度解读**: 生成2000-3000字中英双语深度注疏
5. **AI战例对照**: 加入2025年AI博弈实例（如 Groq vs Nvidia、xAI vs OpenAI、Ministral 3边缘部署革命）

## 用户今日心得：

{user_note}

## 输出格式要求：

请以 Markdown 格式输出，包含以下部分：

# [书名]·[篇章名] - AI时代注疏

**日期**: {datetime.now().strftime('%Y年%m月%d日')}
**分类**: [孙子/资治/圣经] > [具体篇章]
**引擎**: Ministral-3 本地推理

## 一、原文引用

[中文原文]

[English Translation]

## 二、历代注疏精选

[引用2-3条经典注疏，注明出处]

## 三、深度解读

[你的2000字分析，中英双语]

## 四、2025 AI时代对照

[映射到当前AI战局的具体案例]

### 战例分析
- **Ministral 3革命**: Apache 2.0开源，边缘部署最佳性价比
- **Groq vs Nvidia**: LPU专用推理vs通用GPU
- **xAI vs OpenAI**: Grok开源vs GPT闭源
- **边缘ASIC**: 手机/汽车/卫星AI推理的未来

## 五、启示与思考

[总结性思考，300字]

---

**标签**: #三书注疏 #AI时代 #Ministral3本地

请直接输出完整的Markdown内容。"""

    result = call_ollama(prompt, model)

    # 元数据
    metadata = {
        "date": datetime.now().isoformat(),
        "model": model,
        "engine": "Ollama (Local)",
        "user_note_length": len(user_note),
        "output_length": len(result)
    }

    return result, metadata


def save_annotation(content: str, metadata: dict) -> Path:
    """保存注疏到文件"""

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{timestamp}-ai-annotation-local.md"
    filepath = ANNOTATIONS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n---\n\n")
        f.write(f"<!-- Metadata: {json.dumps(metadata, ensure_ascii=False)} -->\n")

    return filepath


def save_draft(user_note: str) -> Path:
    """保存原始心得"""

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{timestamp}-draft.md"
    filepath = DRAFTS_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 每日三书心得 - {timestamp}\n\n")
        f.write(user_note)
        f.write("\n")

    return filepath


def main():
    print("=" * 70)
    print("  三书AI注疏系统 - Ministral-3 本地引擎 (Ollama)")
    print("  《孙子兵法》《资治通鉴》《圣经》AI时代解读")
    print("  🚀 完全本地运行，数据不出你的机器，完全免费")
    print("=" * 70)
    print()

    # 检查Ollama
    if not check_ollama():
        print("❌ Ollama未安装或未运行")
        print()
        print("快速安装:")
        print("  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh")
        print("  或访问: https://ollama.com/download")
        print()
        print("安装后运行:")
        print("  ollama pull ministral-3:8b")
        print()
        sys.exit(1)

    # 检查模型
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

    # 获取用户输入
    if len(sys.argv) > 1:
        user_note = " ".join(sys.argv[1:])
    else:
        print("请输入今天的三书心得（300-800字）：")
        print("（可以包含你对《孙子》《资治》《圣经》任意一部的思考）")
        print()
        user_note = input("> ")

    if not user_note.strip():
        print("❌ 心得不能为空")
        sys.exit(1)

    print()
    print(f"📝 收到心得 ({len(user_note)} 字)")
    print()

    # 保存原始心得
    draft_path = save_draft(user_note)
    print(f"✅ 原始心得已保存: {draft_path.relative_to(REPO_ROOT)}")
    print()

    # 生成AI注疏
    annotation, metadata = generate_annotation(user_note, model)

    # 保存注疏
    annotation_path = save_annotation(annotation, metadata)
    print()
    print(f"✅ AI注疏已生成: {annotation_path.relative_to(REPO_ROOT)}")
    print(f"📊 输出长度: {metadata['output_length']} 字")
    print(f"🤖 模型: {metadata['model']} (本地)")
    print()

    # 预览
    print("=" * 70)
    print("📖 注疏预览（前500字）：")
    print("=" * 70)
    preview = annotation[:500]
    print(preview)
    if len(annotation) > 500:
        print("\n... (省略剩余内容)")
    print()

    # 询问是否提交
    print("=" * 70)
    print("💾 是否自动提交到Git？(y/n)")
    choice = input("> ").lower()

    if choice in ['y', 'yes', '是']:
        print()
        print("🚀 正在提交...")

        os.chdir(REPO_ROOT)
        os.system("git add three-books-ai-annotations/ books-ai-publishing/drafts/")

        commit_msg = f"docs: Daily annotation {datetime.now().strftime('%Y-%m-%d')} (Ministral-3 local)"
        os.system(f'git commit -m "{commit_msg}"')

        print()
        print("是否推送到GitHub？(y/n)")
        push_choice = input("> ").lower()

        if push_choice in ['y', 'yes', '是']:
            os.system("git push origin main")
            print()
            print("✅ 已推送到GitHub!")
        else:
            print()
            print("✅ 已提交到本地，稍后可运行 'git push' 推送")
    else:
        print()
        print("✅ 已生成注疏文件，可稍后手动提交")

    print()
    print("=" * 70)
    print("🎉 完成！继续保持每日心得，7年后见证传世之作。")
    print("=" * 70)


if __name__ == "__main__":
    main()
