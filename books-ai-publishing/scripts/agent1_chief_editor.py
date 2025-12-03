#!/usr/bin/env python3
"""
Agent 1: 总编辑（Chief Editor）

功能：
1. 读取用户的每日心得
2. 判断属于哪本书（《孙子兵法》/《资治通鉴》/《圣经》）
3. 确定具体的章节或篇目
4. 提取核心主题标签
5. 创建结构化的markdown文件

作者: Jim Xiao
版本: 1.0
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

# 导入核心库
from agent_core import AgentCore, PROJECT_ROOT, ANNOTATIONS_DIR


class ChiefEditor(AgentCore):
    """总编辑Agent"""

    def __init__(self, api_key: str = None):
        super().__init__("Chief Editor", api_key)

    def classify_and_structure(self, draft_content: str, draft_file: str) -> Tuple[Dict, str]:
        """
        分类心得并创建结构化文档

        Args:
            draft_content: 用户心得内容
            draft_file: draft文件名

        Returns:
            (metadata, structured_content) 元组
        """
        self.log("开始分类和结构化...")

        # 读取Agent配置
        agent_config = self.read_agent_config(1)

        # 构建提示词
        prompt = f"""你是三书AI注疏项目的总编辑。请阅读以下用户心得，完成分类和结构化任务。

## 用户心得

```markdown
{draft_content}
```

## 任务

### 1. 判断属于哪本书
从以下三本书中选择一本：
- 《孙子兵法》（13篇）
- 《资治通鉴》（294卷）
- 《圣经》（1189章）

### 2. 确定具体章节
根据内容中提到的原文或主题，确定具体的章节名称。

### 3. 提取主题标签
提取1-3个核心主题标签（如：战略隐藏、时机选择、AI竞赛等）

### 4. 识别AI战例
提取文中提到的AI公司/事件（如：OpenAI Q*、xAI Grok等）

## 输出格式

请以JSON格式输出（不要包含markdown代码块标记）：

{{
  "book": "sunzi|zizhi|bible",
  "book_name": "书名",
  "chapter": "章节名称",
  "chapter_file": "文件名（如01-始计篇.md）",
  "themes": ["主题1", "主题2", "主题3"],
  "ai_parallels": ["AI战例1", "AI战例2"],
  "summary": "一句话总结核心洞见"
}}

注意：
- book必须是sunzi/zizhi/bible之一
- chapter_file格式示例：孙子兵法用"01-始计篇.md"，资治通鉴用"卷001-周纪一.md"
- 只输出JSON，不要其他文字
"""

        # 调用Claude
        response = self.call_claude(
            prompt,
            system="你是专业的文献分类和结构化专家，擅长《孙子兵法》《资治通鉴》《圣经》三部经典。",
            max_tokens=2000,
            temperature=0.3  # 分类任务用低温度，确保准确性
        )

        # 解析JSON响应
        import json

        try:
            # 清理响应（移除可能的markdown代码块标记）
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            classification = json.loads(response)
        except json.JSONDecodeError as e:
            self.log(f"JSON解析失败: {e}", "ERROR")
            self.log(f"Claude返回: {response}", "ERROR")
            raise

        # 提取元数据
        metadata = {
            "book": classification["book_name"],
            "chapter": classification["chapter"],
            "date": self.extract_date_from_filename(draft_file),
            "themes": classification["themes"],
            "ai_parallels": classification.get("ai_parallels", []),
            "original_draft": f"drafts/{draft_file}",
            "status": "待完善",
            "version": "0.1"
        }

        # 生成结构化内容
        structured_content = self._generate_structured_content(
            draft_content,
            classification,
            draft_file
        )

        self.log(f"分类完成: {classification['book_name']} - {classification['chapter']}", "SUCCESS")

        return metadata, structured_content, classification

    def _generate_structured_content(
        self,
        draft_content: str,
        classification: Dict,
        draft_file: str
    ) -> str:
        """生成结构化Markdown内容"""

        word_count = self.count_words(draft_content)
        date = self.extract_date_from_filename(draft_file)

        content = f"""# {classification["chapter"]}·AI时代解读（{date}）

> **核心洞见**：{classification["summary"]}
> **关联项目**：3.5-bit量化 + SPARK证明的战略定位
> **阅读时长**：{max(5, word_count // 200)}分钟

---

## 目录
1. [原文与用户心得](#原文与用户心得)
2. [学术注疏](#学术注疏)（Agent 2待添加）
3. [AI时代战略解读](#ai时代战略解读)（Agent 3待添加）
4. [行动建议](#行动建议)
5. [参考文献](#参考文献)

---

## 原文与用户心得

### 《{classification["book_name"]}·{classification["chapter"]}》原文
> （待Agent 2添加完整原文）

### {date} 读书心得

{draft_content}

---

## 编辑批注（Agent 1完成）

- **核心洞见**：{classification["summary"]}
- **主题标签**：{", ".join(classification["themes"])}
- **AI战例**：{", ".join(classification.get("ai_parallels", []))}

**分类结果**：
- 书籍：{classification["book_name"]}
- 章节：{classification["chapter"]}

## 待后续Agent处理
- [ ] Agent 2: 添加历代注疏和中英双语对照
- [ ] Agent 3: 深入分析AI时代战略对照和产业推演
- [ ] Agent 4: 校对和润色

---

**Agent 1处理信息**：
- 处理时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 分类准确度：待验证
- 输入字数：{word_count}字
- 结构化完成：✅
"""

        return content

    def process_draft(self, draft_file: str) -> Path:
        """
        处理一个draft文件

        Args:
            draft_file: draft文件名

        Returns:
            输出文件路径
        """
        self.log(f"开始处理: {draft_file}")

        # 读取draft
        draft_content, _ = self.read_draft(draft_file)
        word_count = self.count_words(draft_content)
        self.log(f"读取成功: {word_count}字")

        # 分类和结构化
        metadata, structured_content, classification = self.classify_and_structure(
            draft_content,
            draft_file
        )

        # 写入注疏文件
        output_path = self.write_annotation(
            book=classification["book"],
            chapter=classification["chapter_file"],
            content=structured_content,
            metadata=metadata
        )

        self.log(f"输出文件: {output_path}", "SUCCESS")

        return output_path


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Agent 1: 总编辑 - 分类和结构化用户心得"
    )
    parser.add_argument(
        "draft_file",
        help="要处理的draft文件名（如：2025-12-03.md）"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API密钥（可选，默认从环境变量读取）"
    )

    args = parser.parse_args()

    try:
        # 创建Agent并处理
        agent = ChiefEditor(api_key=args.api_key)
        output_path = agent.process_draft(args.draft_file)

        print(f"\n✅ 处理完成！")
        print(f"📄 输出文件: {output_path}")
        print(f"\n下一步:")
        print(f"  python scripts/agent2_annotator.py {output_path}")

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
