#!/bin/bash

##############################################################################
# 三书AI注疏每日工作流 - Claude Code Task版本
#
# 这个版本直接使用Claude Code的Task工具，无需额外配置
# 适合Claude Code用户快速上手
#
# 用法:
#   ./scripts/daily-flow-claude.sh                    # 处理今日心得
#   ./scripts/daily-flow-claude.sh 2025-12-03.md      # 处理指定日期心得
#
# 作者: Jim Xiao
# 版本: 1.0 (Claude Code Task版)
# 最后更新: 2025-12-03
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_agent() { echo -e "${BLUE}🤖 $1${NC}"; }

# 确定要处理的文件
DRAFT_FILE="${1:-$(date +%Y-%m-%d).md}"
DRAFT_PATH="$PROJECT_ROOT/drafts/$DRAFT_FILE"

if [ ! -f "$DRAFT_PATH" ]; then
    print_error "文件不存在: $DRAFT_PATH"
    exit 1
fi

print_success "找到draft文件: $DRAFT_FILE"

# 读取文件内容
DRAFT_CONTENT=$(cat "$DRAFT_PATH")
WORD_COUNT=$(echo "$DRAFT_CONTENT" | wc -m)

print_info "字数统计: $WORD_COUNT 字"
echo ""

# ============================================================================
# 使用Claude Code直接处理
# ============================================================================

print_agent "启动Claude Code处理流程..."
print_info "这将在当前Claude Code会话中运行5个Agent任务"
echo ""

# 创建临时提示文件
TEMP_PROMPT="$PROJECT_ROOT/.temp_agent_prompt.md"

cat > "$TEMP_PROMPT" << 'PROMPT_END'
# AI注疏自动化任务

你现在是三书AI注疏出版系统的执行器。请按顺序完成以下5个Agent任务：

## 📥 输入内容

以下是用户的每日心得：

```markdown
DRAFT_CONTENT_PLACEHOLDER
```

## 🤖 任务清单

### Agent 1: 总编辑（10秒）
**配置文件**: `books-ai-publishing/agents/01-chief-editor.md`

**任务**：
1. 判断这段心得属于哪本书（《孙子兵法》/《资治通鉴》/《圣经》）
2. 确定具体的章节或篇目
3. 提取1-3个核心主题标签
4. 在 `books-ai-publishing/annotations/[book]/` 创建结构化markdown文件

**输出文件格式**：
```markdown
---
book: 孙子兵法
chapter: 始计篇
date: YYYY-MM-DD
themes: [战略隐藏, AI竞赛, 时机选择]
ai_parallels: [OpenAI Q*, xAI Grok]
original_draft: drafts/YYYY-MM-DD.md
---

# 始计篇 - YYYY-MM-DD 读书笔记

## 原文
> [从用户心得中提取的原文引用]

## 用户心得（原文）
[复制用户的完整心得]

## 编辑批注
- **核心洞见**: [总结]
- **历史对照**: [AI战例]
- **技术对照**: [技术方案]
- **个人关联**: [7年大业]

## 待后续Agent处理
- [ ] Agent 2: 添加历代注疏
- [ ] Agent 3: AI战略分析
- [ ] Agent 4: 校对润色
```

---

### Agent 2: 注疏师（20秒）
**配置文件**: `books-ai-publishing/agents/02-annotator.md`

**任务**：
在Agent 1创建的文件基础上，添加：
1. **原文全文及注音**（找到完整段落）
2. **历代注家解读**（至少3-5位）
3. **中英双语对照**（至少2个权威英译本）
4. **现代学术研究**（至少1篇2020年后论文）
5. **考证说明**（关键词、版本、断句争议）

**注意**：
- 孙子兵法：引用曹操、李筌、杜牧、梅尧臣、王皙等注家
- 英译本：Griffith (1963), Sawyer (1994), Ames (1993)
- 资治通鉴：引用胡三省音注
- 圣经：引用Matthew Henry, Keil & Delitzsch等

---

### Agent 3: AI战略家（30秒）
**配置文件**: `books-ai-publishing/agents/03-ai-strategist.md`

**任务**：
在同一文件添加：
1. **2025年AI产业地图**（四大势力、核心博弈）
2. **古代智慧在AI竞赛中的显现**（至少3个真实战例）
3. **用户7年大业的战略定位**
4. **未来3年AI产业推演**
5. **可执行的行动建议**（带具体时间节点）

**真实AI战例来源**（2024-2025）：
- OpenAI: Q*事件、GPT-4o、Sam Altman复职、估值1570亿
- Anthropic: Constitutional AI、Claude 3.5、融资75亿
- xAI: Grok开源、Colossus超算100K H100
- NVIDIA: H100饥饿营销、GB200发布
- Meta: Llama 3开源、405B模型

**连接用户项目**：
- 3.5-bit量化（vs 大厂堆参数）
- SPARK + Lean 4证明（247条性质）
- 时间窗口：12-24个月（2025-2026）
- 目标：NeurIPS 2026、FAA/DoD认证

---

### Agent 4: 校对神（10秒）
**配置文件**: `books-ai-publishing/agents/04-proofreader.md`

**任务**：
1. **错别字检查**（中英文）
2. **引文核对**（所有注家姓名、英译本译者、年份）
3. **逻辑检查**（论证链条完整性）
4. **格式统一**（Markdown规范、学术引用格式）
5. **生成校对后的终稿**到 `books-ai-publishing/proofread/`

**输出**：
- 错误修正列表
- 最终字数统计
- 出版就绪标记

---

### Agent 5: 生成摘要（5秒）

创建 `books-ai-publishing/proofread/daily-summary-YYYY-MM-DD.md`：

```markdown
# 每日AI注疏处理摘要

**日期**: YYYY-MM-DD
**输入文件**: DRAFT_FILE_PLACEHOLDER
**字数**: WORD_COUNT_PLACEHOLDER 字

## 处理流程
- ✅ Agent 1: 总编辑 - 分类到《书名》《章节》
- ✅ Agent 2: 注疏师 - 添加X位注家解读 + Y个英译本
- ✅ Agent 3: AI战略家 - 分析Z个AI战例
- ✅ Agent 4: 校对神 - 修正M处错误

## 输出文件
- 注疏文件: annotations/[book]/[chapter].md (最终字数)
- 校对版本: proofread/YYYY-MM-DD.md

## 核心洞见
[一句话总结]

## 下一步
- [ ] 继续明天的阅读
- [ ] 每周六自动生成周论
```

---

## ⚠️ 重要提示

1. **按顺序执行**：必须先完成Agent 1，才能运行Agent 2-4
2. **使用Read工具**：读取Agent配置文件获取详细指导
3. **使用Write/Edit工具**：创建和编辑输出文件
4. **真实引用**：所有注家、译者、论文必须真实存在
5. **完整性**：最终输出应该是5000+字的完整文档

## 🎯 成功标准

✅ Agent 1创建的文件包含完整YAML front matter和结构
✅ Agent 2添加了至少3位注家 + 2个英译本
✅ Agent 3包含至少3个2024-2025真实AI战例
✅ Agent 4修正了所有错误并生成终稿
✅ Agent 5生成了处理摘要

---

现在开始执行！
PROMPT_END

# 替换占位符
sed -i.bak "s|DRAFT_CONTENT_PLACEHOLDER|$DRAFT_CONTENT|g" "$TEMP_PROMPT"
sed -i.bak "s|DRAFT_FILE_PLACEHOLDER|$DRAFT_FILE|g" "$TEMP_PROMPT"
sed -i.bak "s|WORD_COUNT_PLACEHOLDER|$WORD_COUNT|g" "$TEMP_PROMPT"

print_info "提示文件已创建: $TEMP_PROMPT"
echo ""

print_agent "==============================================="
print_agent "请在Claude Code中运行以下命令："
print_agent "==============================================="
echo ""
echo "claude code --prompt @$TEMP_PROMPT"
echo ""
print_agent "或者直接复制以下内容到Claude Code："
echo ""
cat "$TEMP_PROMPT"
echo ""

print_info "脚本已准备好提示文件。"
print_info "由于这是在bash脚本中，无法直接调用Claude Code的交互会话。"
print_info "请在Claude Code会话中手动运行上述提示。"
echo ""

print_success "准备完成！等待你在Claude Code中执行。"
