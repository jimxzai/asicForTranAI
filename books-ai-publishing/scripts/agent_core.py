#!/usr/bin/env python3
"""
三书AI注疏核心工具库

这个模块提供所有Agent的核心功能：
- Claude API调用
- Markdown文件读写
- YAML front matter处理
- 日志和错误处理

作者: Jim Xiao
版本: 1.0
"""

import os
import sys
import re
import yaml
import frontmatter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DRAFTS_DIR = PROJECT_ROOT / "drafts"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"
AI_PARALLELS_DIR = PROJECT_ROOT / "ai-parallels"
PROOFREAD_DIR = PROJECT_ROOT / "proofread"
AGENTS_DIR = PROJECT_ROOT / "agents"


class AgentCore:
    """Agent核心类 - 所有Agent的基类"""

    def __init__(self, agent_name: str, api_key: Optional[str] = None):
        """
        初始化Agent

        Args:
            agent_name: Agent名称（如"Chief Editor", "Annotator"等）
            api_key: Anthropic API密钥（如果不提供，从环境变量读取）
        """
        self.agent_name = agent_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API密钥未设置！\n"
                "请设置环境变量：export ANTHROPIC_API_KEY='your-key'\n"
                "或在初始化时传入：AgentCore(api_key='your-key')"
            )

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"  # 使用最新的Sonnet 4.5

    def call_claude(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 8000,
        temperature: float = 0.7
    ) -> str:
        """
        调用Claude API

        Args:
            prompt: 用户提示词
            system: 系统提示词（可选）
            max_tokens: 最大生成token数
            temperature: 温度参数（0-1，越高越随机）

        Returns:
            Claude的回复文本
        """
        try:
            message_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            if system:
                message_params["system"] = system

            response = self.client.messages.create(**message_params)

            # 提取文本内容
            return response.content[0].text

        except Exception as e:
            print(f"❌ Claude API调用失败: {e}")
            raise

    def read_draft(self, draft_file: str) -> Tuple[str, Dict]:
        """
        读取draft文件

        Args:
            draft_file: draft文件名（如"2025-12-03.md"）

        Returns:
            (content, metadata) 元组
            - content: 文件内容（字符串）
            - metadata: 元数据字典（如果有YAML front matter）
        """
        draft_path = DRAFTS_DIR / draft_file

        if not draft_path.exists():
            raise FileNotFoundError(f"Draft文件不存在: {draft_path}")

        # 使用python-frontmatter解析
        post = frontmatter.load(draft_path)

        return post.content, post.metadata

    def write_annotation(
        self,
        book: str,
        chapter: str,
        content: str,
        metadata: Dict
    ) -> Path:
        """
        写入注疏文件

        Args:
            book: 书籍名称（sunzi/zizhi/bible）
            chapter: 章节名称（如"01-始计篇.md"）
            content: 文件内容
            metadata: YAML front matter元数据

        Returns:
            写入的文件路径
        """
        # 创建目录
        book_dir = ANNOTATIONS_DIR / book
        book_dir.mkdir(parents=True, exist_ok=True)

        # 写入文件（带YAML front matter）
        output_path = book_dir / chapter

        post = frontmatter.Post(content, **metadata)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))

        return output_path

    def read_agent_config(self, agent_number: int) -> str:
        """
        读取Agent配置文件

        Args:
            agent_number: Agent编号（1-5）

        Returns:
            Agent配置内容（Markdown格式）
        """
        config_files = {
            1: "01-chief-editor.md",
            2: "02-annotator.md",
            3: "03-ai-strategist.md",
            4: "04-proofreader.md",
            5: "05-publisher.md"
        }

        config_file = AGENTS_DIR / config_files[agent_number]

        if not config_file.exists():
            raise FileNotFoundError(f"Agent配置文件不存在: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            return f.read()

    def classify_book(self, content: str) -> Tuple[str, str]:
        """
        分类心得属于哪本书和哪一章节

        Args:
            content: 用户心得内容

        Returns:
            (book, chapter) 元组
            - book: "sunzi" / "zizhi" / "bible"
            - chapter: 章节名称
        """
        # 简单的关键词匹配（实际Agent 1会用Claude做更智能的分类）
        if "孙子" in content or "兵法" in content:
            return "sunzi", "未知章节"
        elif "资治通鉴" in content or "通鉴" in content:
            return "zizhi", "未知卷"
        elif "圣经" in content or "Bible" in content:
            return "bible", "未知章"
        else:
            # 默认使用Claude分类
            return "unknown", "unknown"

    def extract_date_from_filename(self, filename: str) -> str:
        """
        从文件名提取日期

        Args:
            filename: 文件名（如"2025-12-03.md"）

        Returns:
            日期字符串（"2025-12-03"）
        """
        match = re.match(r'(\d{4}-\d{2}-\d{2})', filename)
        if match:
            return match.group(1)
        else:
            return datetime.now().strftime("%Y-%m-%d")

    def count_words(self, text: str) -> int:
        """
        统计中文字数（简单实现）

        Args:
            text: 文本内容

        Returns:
            字数
        """
        # 移除markdown标记
        text = re.sub(r'[#*`\[\]()]', '', text)
        # 统计中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文单词
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))

        return chinese_chars + english_words

    def log(self, message: str, level: str = "INFO"):
        """
        日志输出

        Args:
            message: 日志消息
            level: 日志级别（INFO/WARNING/ERROR）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }.get(level, "📝")

        print(f"[{timestamp}] {prefix} [{self.agent_name}] {message}")


def load_config(config_file: str = "config.yaml") -> Dict:
    """
    加载配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    config_path = PROJECT_ROOT / config_file

    if not config_path.exists():
        # 返回默认配置
        return {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 8000,
            "temperature": 0.7,
            "books": {
                "sunzi": {
                    "name": "孙子兵法",
                    "chapters": 13
                },
                "zizhi": {
                    "name": "资治通鉴",
                    "chapters": 294
                },
                "bible": {
                    "name": "圣经",
                    "chapters": 1189
                }
            }
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试AgentCore...")

    try:
        agent = AgentCore("Test Agent")
        agent.log("初始化成功", "SUCCESS")

        # 测试读取draft
        draft_file = "2025-12-03-example.md"
        if (DRAFTS_DIR / draft_file).exists():
            content, metadata = agent.read_draft(draft_file)
            agent.log(f"读取draft成功: {len(content)}字", "SUCCESS")

            # 测试分类
            book, chapter = agent.classify_book(content)
            agent.log(f"分类结果: {book} - {chapter}", "INFO")

        agent.log("所有测试通过！", "SUCCESS")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)
