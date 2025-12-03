#!/bin/bash

##############################################################################
# 三书AI注疏出版系统 - 环境设置脚本
#
# 用法:
#   ./scripts/setup.sh
#
# 功能:
#   1. 检查Python版本
#   2. 创建虚拟环境
#   3. 安装依赖包
#   4. 配置环境变量
#   5. 验证安装
#
# 作者: Jim Xiao
# 版本: 1.0
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

echo "=================================================="
echo "  三书AI注疏出版系统 - 环境设置"
echo "=================================================="
echo ""

# 1. 检查Python版本
print_info "检查Python版本..."
PYTHON_CMD="python3"

if ! command -v $PYTHON_CMD &> /dev/null; then
    print_error "Python 3未安装！请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
print_success "Python版本: $PYTHON_VERSION"

# 检查版本号
MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
    print_error "Python版本过低！需要Python 3.8+，当前版本：$PYTHON_VERSION"
    exit 1
fi

echo ""

# 2. 创建虚拟环境
print_info "创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "虚拟环境创建成功"
else
    print_warning "虚拟环境已存在，跳过创建"
fi

echo ""

# 3. 激活虚拟环境
print_info "激活虚拟环境..."
source venv/bin/activate
print_success "虚拟环境已激活"

echo ""

# 4. 升级pip
print_info "升级pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip已升级到最新版本"

echo ""

# 5. 安装依赖包
print_info "安装依赖包..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    print_success "所有依赖包安装完成"
else
    print_error "依赖包安装失败"
    exit 1
fi

echo ""

# 6. 配置环境变量
print_info "配置环境变量..."

if [ -f ".env" ]; then
    print_warning ".env文件已存在"
else
    cat > .env << 'EOF'
# Anthropic API密钥
ANTHROPIC_API_KEY=your-api-key-here

# 可选：NotebookLM API（如果未来有的话）
# NOTEBOOKLM_API_KEY=your-notebooklm-api-key

# 可选：其他API密钥
# PERPLEXITY_API_KEY=your-perplexity-api-key
# DEEPL_API_KEY=your-deepl-api-key
EOF
    print_success "已创建.env模板文件"
    print_warning "请编辑.env文件，填入你的API密钥"
fi

echo ""

# 7. 验证安装
print_info "验证安装..."

echo "测试Python导入..."
python3 << 'PYTHON_TEST'
try:
    import anthropic
    import yaml
    import frontmatter
    print("✅ 所有必需包导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    exit(1)
PYTHON_TEST

if [ $? -ne 0 ]; then
    print_error "安装验证失败"
    exit 1
fi

echo ""

# 8. 测试Agent核心
print_info "测试Agent核心..."
cd "$(dirname "$0")/.."
python3 scripts/agent_core.py

if [ $? -eq 0 ]; then
    print_success "Agent核心测试通过"
else
    print_error "Agent核心测试失败"
    exit 1
fi

echo ""

# 9. 完成
print_success "=========================================="
print_success "  环境设置完成！"
print_success "=========================================="
echo ""

print_info "下一步："
echo "  1. 编辑.env文件，填入你的Anthropic API密钥"
echo "  2. 激活虚拟环境: source venv/bin/activate"
echo "  3. 运行测试: ./scripts/daily-flow.sh --dry-run 2025-12-03-example.md"
echo "  4. 开始写你的第一篇心得！"
echo ""

print_info "快速开始："
echo "  vim drafts/\$(date +%Y-%m-%d).md"
echo "  python scripts/agent1_chief_editor.py \$(date +%Y-%m-%d).md"
echo ""

print_success "祝你7年大业顺利！🚀"
