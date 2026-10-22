#!/bin/bash
# 极简人生：每天4小时改变世界
# 2025-12-03 → 2032-12-03: 7年，4本书，航空级AI推理引擎

set -e

REPO_ROOT="/Users/jimxiao/dev/2025/AI2025/asicForTranAI"
cd "$REPO_ROOT"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║           🌅 极简人生：每天4小时                                ║
║                                                                ║
║   07:00-09:30: 读书+注疏+代码 (2.5小时)                       ║
║   其余时间: 钓鱼、陪家人、喝茶、睡觉                           ║
║                                                                ║
║   7年后: 4本书 + 航空级AI + 周工作3天 + $200万/年             ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 检查时间
CURRENT_HOUR=$(date +%H)
if [ "$CURRENT_HOUR" -lt 7 ] || [ "$CURRENT_HOUR" -gt 10 ]; then
    echo -e "${YELLOW}⚠️  当前时间: $(date +%H:%M)${NC}"
    echo -e "${YELLOW}   建议在 07:00-09:30 之间运行（最佳专注时段）${NC}"
    echo ""
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  步骤1/3: 读三书原文 (30分钟)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "📖 今天读什么？"
echo ""
echo "  1) 《孙子兵法》"
echo "  2) 《资治通鉴》"
echo "  3) 《圣经》"
echo "  4) 跳过（已经读过）"
echo ""
read -p "选择 (1-4): " book_choice

case $book_choice in
    1) BOOK="孙子兵法" ;;
    2) BOOK="资治通鉴" ;;
    3) BOOK="圣经" ;;
    4)
        echo ""
        echo -e "${GREEN}✅ 跳过读书环节${NC}"
        BOOK="已读"
        ;;
    *)
        echo "无效选择，默认《孙子兵法》"
        BOOK="孙子兵法"
        ;;
esac

if [ "$BOOK" != "已读" ]; then
    echo ""
    echo -e "${CYAN}📚 正在阅读《${BOOK}》...${NC}"
    echo ""
    echo "（现在拿起书，读30分钟，或者听有声书）"
    echo ""
    echo "按 Enter 完成阅读..."
    read
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  步骤2/3: 写心得 + AI注疏 (30分钟)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "✍️  请输入你的心得（300-500字）："
echo "   提示: 关于今天读的内容，结合2025年AI战局的思考"
echo ""
echo "（输入完成后按 Ctrl+D）"
echo ""

# 读取多行输入
NOTE=$(cat)

if [ -z "$NOTE" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  心得为空，使用示例心得${NC}"
    NOTE="今天读《孙子·始计篇》'兵者，国之大事'。想到2025年AI芯片战，Groq专注LPU就像孙子说的'不可不察'——生死存亡之道。"
fi

NOTE_LEN=$(echo "$NOTE" | wc -c)
echo ""
echo -e "${GREEN}✅ 收到心得 (${NOTE_LEN} 字)${NC}"
echo ""

# 选择引擎
echo "🤖 选择AI注疏引擎："
echo ""
echo "  1) Ministral-3 本地 (Ollama) - 推荐！完全免费，数据不出机器"
echo "  2) Llama-3.3-70B (Groq Cloud) - 云端，速度更快"
echo ""
read -p "选择 (1-2, 默认1): " engine_choice

case $engine_choice in
    2)
        echo ""
        if [ -z "$GROQ_API_KEY" ]; then
            echo -e "${RED}❌ GROQ_API_KEY 未设置${NC}"
            echo ""
            echo "请先运行: export GROQ_API_KEY=\"gsk_...\""
            echo "或选择本地Ollama引擎"
            exit 1
        fi
        echo -e "${CYAN}🚀 使用 Llama-3.3-70B @ Groq Cloud${NC}"
        python3 agents/llama33-70b/daily-three-books-agent.py "$NOTE"
        ;;
    *)
        echo ""
        echo -e "${CYAN}🚀 使用 Ministral-3 @ Ollama 本地${NC}"

        # 检查是否有ministral-3
        if ! ollama list | grep -q "ministral-3"; then
            echo ""
            echo -e "${YELLOW}⚠️  Ministral-3 未安装${NC}"
            echo ""
            echo "正在下载 ministral-3:8b (约5.2GB, 首次运行需要)..."
            echo ""
            ollama pull ministral-3:8b
            echo ""
            echo -e "${GREEN}✅ Ministral-3 下载完成${NC}"
            echo ""
        fi

        python3 agents/ministral-3-local/daily-annotation-ollama.py "$NOTE"
        ;;
esac

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  步骤3/3: 提交到GitHub (15秒)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 自动commit和push（如果agent里没做的话）
if git status --porcelain | grep -q "three-books-ai-annotations\|drafts"; then
    echo "📤 正在提交到GitHub..."
    git add three-books-ai-annotations/ books-ai-publishing/drafts/
    git commit -m "docs: Daily annotation $(date +%Y-%m-%d)"
    git push origin main
    echo ""
    echo -e "${GREEN}✅ 已推送到 https://github.com/jimxzai/asicForTranAI${NC}"
else
    echo -e "${GREEN}✅ 无新内容或已提交${NC}"
fi

echo ""
echo -e "${CYAN}"
cat << 'EOF'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  🎉 今日任务完成！                                             ║
║                                                                ║
║  ✅ 读书: 30分钟                                               ║
║  ✅ 心得: 300-500字                                            ║
║  ✅ AI注疏: 2500字自动生成                                      ║
║  ✅ 推送: GitHub已更新                                         ║
║                                                                ║
║  今天总耗时: ~1小时                                            ║
║  剩余时间: 做你喜欢的事！                                      ║
║                                                                ║
║  7年倒计时: 还剩 2554 天                                       ║
║  明天见！🫶                                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# 显示进度
ANNOTATION_COUNT=$(find three-books-ai-annotations -name "*.md" | wc -l | tr -d ' ')
TOTAL_WORDS=$((ANNOTATION_COUNT * 2500))

echo ""
echo "📊 累计进度:"
echo "   注疏篇数: ${ANNOTATION_COUNT}"
echo "   累计字数: ~${TOTAL_WORDS} 字"
echo "   距离第一本书 (30万字): $(echo "scale=1; $TOTAL_WORDS / 300000 * 100" | bc)%"
echo ""

# 周六提醒
if [ "$(date +%u)" -eq 6 ]; then
    echo -e "${YELLOW}🎯 今天是周六！GitHub Actions会自动生成本周周论PDF${NC}"
    echo ""
fi

echo "🌐 查看仓库: https://github.com/jimxzai/asicForTranAI"
echo ""
