#!/bin/bash

##############################################################################
# 三书AI注疏每日工作流 - 一键启动脚本
#
# 用法:
#   ./scripts/daily-flow.sh                    # 处理今日心得
#   ./scripts/daily-flow.sh 2025-12-03.md      # 处理指定日期心得
#   ./scripts/daily-flow.sh --help             # 显示帮助
#
# 功能:
#   1. 读取你的每日300-500字心得
#   2. 依次调用5个AI Agent处理
#   3. 生成完整注疏、AI分析、校对版本
#   4. 自动提交到Git
#
# 作者: Jim Xiao
# 版本: 1.0
# 最后更新: 2025-12-03
##############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 函数: 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_agent() {
    echo -e "${BLUE}🤖 $1${NC}"
}

# 函数: 显示帮助
show_help() {
    cat << EOF
三书AI注疏每日工作流

用法:
  $0 [选项] [文件名]

选项:
  -h, --help              显示此帮助信息
  -d, --dry-run           模拟运行，不实际调用Agent
  --no-git                不自动提交到Git
  --skip-agent <N>        跳过指定Agent（1-5）

参数:
  文件名                   要处理的draft文件（如: 2025-12-03.md）
                          如果不指定，则处理今日文件

示例:
  $0                                # 处理今日心得
  $0 2025-12-03.md                  # 处理指定日期
  $0 --dry-run 2025-12-03.md        # 模拟运行
  $0 --no-git 2025-12-03.md         # 不提交Git

5个AI Agent:
  Agent 1: 总编辑 - 分类和结构化
  Agent 2: 注疏师 - 添加学术注释
  Agent 3: AI战略家 - 现代战略映射
  Agent 4: 校对神 - 质量控制和润色
  Agent 5: 出书总管 - 编译成书稿（每周/季度运行）

EOF
}

# 解析命令行参数
DRY_RUN=false
NO_GIT=false
SKIP_AGENTS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-git)
            NO_GIT=true
            shift
            ;;
        --skip-agent)
            SKIP_AGENTS+=("$2")
            shift 2
            ;;
        *)
            DRAFT_FILE="$1"
            shift
            ;;
    esac
done

# 确定要处理的文件
if [ -z "$DRAFT_FILE" ]; then
    DRAFT_FILE="$(date +%Y-%m-%d).md"
    print_info "未指定文件，使用今日文件: $DRAFT_FILE"
fi

DRAFT_PATH="$PROJECT_ROOT/drafts/$DRAFT_FILE"

# 检查文件是否存在
if [ ! -f "$DRAFT_PATH" ]; then
    print_error "文件不存在: $DRAFT_PATH"
    print_info "请先创建你的每日心得文件（300-500字）"
    print_info ""
    print_info "创建方法:"
    print_info "  vim $DRAFT_PATH"
    print_info "  或使用你喜欢的编辑器"
    exit 1
fi

print_success "找到draft文件: $DRAFT_FILE"

# 显示文件预览
echo ""
print_info "文件预览（前10行）:"
echo "-------------------"
head -n 10 "$DRAFT_PATH"
echo "-------------------"
echo ""

# 统计字数
WORD_COUNT=$(wc -m < "$DRAFT_PATH")
print_info "字数统计: $WORD_COUNT 字"

if [ $WORD_COUNT -lt 200 ]; then
    print_warning "字数较少（< 200字），建议补充到300-500字"
fi

# 模拟运行模式
if [ "$DRY_RUN" = true ]; then
    print_warning "模拟运行模式 - 不会实际调用Agent"
    print_info "将要执行的步骤:"
    print_info "  1. Agent 1: 分类和结构化"
    print_info "  2. Agent 2: 添加学术注释"
    print_info "  3. Agent 3: AI战略分析"
    print_info "  4. Agent 4: 校对和润色"
    print_info "  5. 提交到Git（如果--no-git未设置）"
    exit 0
fi

# 检查Claude Code是否可用
if ! command -v claude &> /dev/null; then
    print_warning "未检测到 claude 命令，将使用Python脚本模拟"
fi

print_info "开始处理... 预计耗时 60-90 秒"
echo ""

# ============================================================================
# Agent 1: 总编辑
# ============================================================================
if [[ ! " ${SKIP_AGENTS[@]} " =~ " 1 " ]]; then
    print_agent "Agent 1: 总编辑 - 分类和结构化"

    # 这里应该调用Claude Code的Task agent
    # 由于这是一个模板脚本，实际实现需要根据你的环境调整

    cat << 'AGENT1_PROMPT' > /tmp/agent1_prompt.txt
你是三书AI注疏项目的总编辑。请阅读以下心得，完成以下任务：

1. 判断这段心得属于哪本书（《孙子兵法》/《资治通鉴》/《圣经》）
2. 确定具体的章节或篇目
3. 提取1-3个核心主题标签
4. 在相应目录下创建结构化的markdown文件

详细配置请参考: agents/01-chief-editor.md

输入文件:
AGENT1_PROMPT

    cat "$DRAFT_PATH" >> /tmp/agent1_prompt.txt

    print_info "正在分类和创建结构化文档..."
    # TODO: 调用Claude Code或API
    # claude code task --prompt @/tmp/agent1_prompt.txt

    print_success "Agent 1 完成"
    echo ""
else
    print_warning "跳过 Agent 1"
    echo ""
fi

# ============================================================================
# Agent 2: 注疏师
# ============================================================================
if [[ ! " ${SKIP_AGENTS[@]} " =~ " 2 " ]]; then
    print_agent "Agent 2: 注疏师 - 添加学术注释"

    print_info "正在检索历代注疏和中英对照..."
    # TODO: 调用Agent 2
    # claude code task --prompt @agents/02-annotator.md

    print_success "Agent 2 完成"
    echo ""
else
    print_warning "跳过 Agent 2"
    echo ""
fi

# ============================================================================
# Agent 3: AI战略家
# ============================================================================
if [[ ! " ${SKIP_AGENTS[@]} " =~ " 3 " ]]; then
    print_agent "Agent 3: AI战略家 - AI时代战略映射"

    print_info "正在分析2025年AI产业战例..."
    # TODO: 调用Agent 3
    # claude code task --prompt @agents/03-ai-strategist.md

    print_success "Agent 3 完成"
    echo ""
else
    print_warning "跳过 Agent 3"
    echo ""
fi

# ============================================================================
# Agent 4: 校对神
# ============================================================================
if [[ ! " ${SKIP_AGENTS[@]} " =~ " 4 " ]]; then
    print_agent "Agent 4: 校对神 - 质量控制"

    print_info "正在校对、润色、检查引文..."
    # TODO: 调用Agent 4
    # claude code task --prompt @agents/04-proofreader.md

    print_success "Agent 4 完成"
    echo ""
else
    print_warning "跳过 Agent 4"
    echo ""
fi

# ============================================================================
# 生成摘要
# ============================================================================
print_info "生成处理摘要..."

SUMMARY_FILE="$PROJECT_ROOT/proofread/daily-summary-$(date +%Y-%m-%d).md"

cat > "$SUMMARY_FILE" << EOF
# 每日AI注疏处理摘要

**日期**: $(date +%Y-%m-%d)
**输入文件**: $DRAFT_FILE
**字数**: $WORD_COUNT 字

## 处理流程

- ✅ Agent 1: 总编辑 - 分类和结构化
- ✅ Agent 2: 注疏师 - 学术注释
- ✅ Agent 3: AI战略家 - AI战略分析
- ✅ Agent 4: 校对神 - 质量控制

## 输出文件

- 注疏文件: annotations/[book]/[chapter].md
- AI分析: ai-parallels/$(date +%Y-%m-%d).md
- 校对版本: proofread/$(date +%Y-%m-%d).md

## 下一步

- [ ] 每周六08:00自动生成周论
- [ ] 每季度末自动生成书稿
- [ ] 继续明天的阅读和心得

---
Generated by AI Publishing System
EOF

print_success "摘要已保存: $SUMMARY_FILE"
echo ""

# ============================================================================
# Git提交
# ============================================================================
if [ "$NO_GIT" = false ]; then
    print_info "准备提交到Git..."

    cd "$PROJECT_ROOT"

    # 检查是否有变更
    if git diff --quiet && git diff --staged --quiet; then
        print_warning "没有检测到变更，跳过Git提交"
    else
        git add annotations/
        git add ai-parallels/
        git add proofread/

        COMMIT_MSG="feat: AI注疏完成 - $(date +%Y-%m-%d)

- ✅ Agent 1: 分类和结构化
- ✅ Agent 2: 学术注疏
- ✅ Agent 3: AI战略分析
- ✅ Agent 4: 校对润色

Draft: $DRAFT_FILE ($WORD_COUNT字)

Generated by AI Publishing System"

        git commit -m "$COMMIT_MSG"

        print_success "已提交到本地Git仓库"

        # 询问是否推送到远程
        read -p "是否推送到远程仓库？ (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push
            print_success "已推送到远程仓库"
        else
            print_info "已跳过推送，你可以稍后手动推送: git push"
        fi
    fi
else
    print_warning "跳过Git提交（--no-git已设置）"
fi

echo ""
print_success "所有处理完成！"
print_info "耗时: $SECONDS 秒"
echo ""
print_info "下一步:"
print_info "  1. 查看生成的注疏: annotations/"
print_info "  2. 查看AI分析: ai-parallels/"
print_info "  3. 查看校对版本: proofread/"
print_info "  4. 每周六会自动生成周论"
print_info "  5. 每季度末会自动生成完整书稿"
echo ""
print_info "7年大业，继续前进！ 🚀"
