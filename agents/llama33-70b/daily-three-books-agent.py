#!/usr/bin/env python3
"""
Daily Three-Books Agent - Llama-3.3-70B-Versatile @ Groq
用法：python daily-three-books-agent.py "今天读《孙子·始计篇》，我的心得是……"

Author: Jim Xiao
Date: 2025-12-03
Model: llama-3.3-70b-versatile (Groq API)
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("⚠️  请先安装 requests: pip install requests")
    sys.exit(1)

# Groq API Configuration
API_KEY = os.environ.get("GROQ_API_KEY", "")  # 从环境变量读取，或填入你的 key
MODEL = "llama-3.3-70b-versatile"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Repository paths
REPO_ROOT = Path(__file__).parent.parent.parent
ANNOTATIONS_DIR = REPO_ROOT / "three-books-ai-annotations"
DRAFTS_DIR = REPO_ROOT / "books-ai-publishing" / "drafts"


def call_llama33(prompt: str, temperature: float = 0.7, max_tokens: int = 8192) -> str:
    """调用 Groq API 的 Llama-3.3-70B 模型"""

    if not API_KEY:
        print("❌ 错误: GROQ_API_KEY 未设置")
        print("请运行: export GROQ_API_KEY='your-key-here'")
        print("或访问: https://console.groq.com/keys")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误: {e}")
        print(f"响应内容: {response.text}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("❌ 请求超时，请重试")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 调用API失败: {e}")
        sys.exit(1)


def generate_annotation(user_note: str) -> tuple[str, dict]:
    """生成AI注疏"""

    prompt = f"""你现在是「AI时代三书注疏大师」，任务是：

1. **自动分类**: 判断用户心得属于《孙子兵法》《资治通鉴》《圣经》中的哪一部、哪一篇/卷/章
2. **检索原文**: 引用相关原文（中英对照）
3. **历代注疏**: 引用经典注疏（如曹操注、杜牧注、胡三省注、马丁·路德注释等）
4. **深度解读**: 生成2000-3000字中英双语深度注疏
5. **AI战例对照**: 加入2025年AI博弈实例（如 Groq vs Nvidia、xAI vs OpenAI、AGI安全博弈）

## 用户今日心得：

{user_note}

## 输出格式要求：

请以 Markdown 格式输出，包含以下部分：

# [书名]·[篇章名] - AI时代注疏

**日期**: {datetime.now().strftime('%Y年%m月%d日')}
**分类**: [孙子/资治/圣经] > [具体篇章]

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
- **Groq vs Nvidia**: [具体对照]
- **xAI vs OpenAI**: [具体对照]
- **AGI安全博弈**: [具体对照]

## 五、启示与思考

[总结性思考，300字]

---

**标签**: #三书注疏 #AI时代 #孙子兵法 #资治通鉴 #圣经

请直接输出完整的Markdown内容。"""

    print("🤖 正在调用 Llama-3.3-70B-Versatile...")
    print(f"📊 模型: {MODEL}")
    print(f"🌐 API: Groq")
    print()

    result = call_llama33(prompt)

    # 尝试提取元数据
    metadata = {
        "date": datetime.now().isoformat(),
        "model": MODEL,
        "user_note_length": len(user_note),
        "output_length": len(result)
    }

    return result, metadata


def save_annotation(content: str, metadata: dict) -> Path:
    """保存注疏到文件"""

    # 创建目录
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y-%m-%d")
    filename = f"{timestamp}-ai-annotation.md"
    filepath = ANNOTATIONS_DIR / filename

    # 保存内容
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n\n---\n\n")
        f.write(f"<!-- Metadata: {json.dumps(metadata, ensure_ascii=False)} -->\n")

    return filepath


def save_draft(user_note: str) -> Path:
    """保存原始心得到drafts目录"""

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
    print("=" * 60)
    print("  三书AI注疏系统 - Llama-3.3-70B Daily Agent")
    print("  《孙子兵法》《资治通鉴》《圣经》AI时代解读")
    print("=" * 60)
    print()

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
    annotation, metadata = generate_annotation(user_note)

    # 保存注疏
    annotation_path = save_annotation(annotation, metadata)
    print()
    print(f"✅ AI注疏已生成: {annotation_path.relative_to(REPO_ROOT)}")
    print(f"📊 输出长度: {metadata['output_length']} 字")
    print()

    # 预览前500字
    print("=" * 60)
    print("📖 注疏预览（前500字）：")
    print("=" * 60)
    preview = annotation[:500]
    print(preview)
    if len(annotation) > 500:
        print("\n... (省略剩余内容)")
    print()

    # 询问是否自动提交
    print("=" * 60)
    print("💾 是否自动提交到Git？(y/n)")
    choice = input("> ").lower()

    if choice in ['y', 'yes', '是']:
        print()
        print("🚀 正在提交...")

        os.chdir(REPO_ROOT)
        os.system("git add three-books-ai-annotations/ books-ai-publishing/drafts/")

        commit_msg = f"docs: Add daily three-books annotation ({datetime.now().strftime('%Y-%m-%d')})"
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
    print("=" * 60)
    print("🎉 完成！继续保持每日心得，7年后见证传世之作。")
    print("=" * 60)


if __name__ == "__main__":
    main()
