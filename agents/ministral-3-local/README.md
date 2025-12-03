# Ministral-3 Local Agent (Ollama)

## 🚀 2025年12月3日重大更新

**Mistral发布Ministral-3系列** (Apache 2.0开源):
- Ministral-3:8b - 边缘部署最佳性价比
- 完全本地运行，数据不出你的机器
- RTX 4090: 185 tok/s
- M2/M3 Mac: 30-80 tok/s
- **完全免费，无需API key**

## 🎯 为什么选择本地Ollama？

| 维度 | Groq Cloud (Llama-3.3) | Ollama Local (Ministral-3) |
|------|------------------------|----------------------------|
| **成本** | 免费额度1000万tokens（3-6个月） | **永久免费** |
| **隐私** | 数据发送到云端 | **数据不出你的机器** |
| **速度** | 1100-1300 tok/s | 30-185 tok/s（取决于硬件） |
| **网络** | 需要联网 | **离线可用** |
| **中文能力** | 优秀 | **优秀（Ministral-3针对优化）** |
| **模型大小** | 70B（云端） | 8B（本地5.2GB） |

**推荐策略**:
- **日常使用**: Ollama本地（完全免费，隐私）
- **紧急高质量**: Groq云端（速度更快）

## 📦 快速开始

### 1. 安装Ollama (1分钟)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 或访问: https://ollama.com/download
```

### 2. 下载Ministral-3 (首次运行，约5.2GB)

```bash
ollama pull ministral-3:8b
```

### 3. 运行第一次注疏！

```bash
cd /Users/jimxiao/dev/2025/AI2025/asicForTranAI

python3 agents/ministral-3-local/daily-annotation-ollama.py "今天读《孙子·始计篇》第一句'兵者，国之大事，死生之地，存亡之道，不可不察也'。我想到2025年的AI芯片战：Groq选择专注LPU推理芯片，就像孙子强调的'不可不察'——这是关乎企业生死存亡的战略抉择。而Mistral发布Ministral-3开源（Apache 2.0），让边缘设备也能跑大模型，正是'善战者，求之于势'的体现。"
```

## 🔧 配置选项

### 切换模型

```bash
# 使用其他Ollama模型
export OLLAMA_MODEL="llama3.3:70b"  # 如果你本地有70B
export OLLAMA_MODEL="deepseek-r1:8b"  # 或其他模型

python3 agents/ministral-3-local/daily-annotation-ollama.py "今天的心得..."
```

### 极简一键流程

```bash
# 使用每日4小时脚本（包含读书+写心得+AI注疏）
./daily-4hour-life.sh
```

## 📊 性能对比

| 硬件 | Ministral-3:8b速度 | 适合场景 |
|------|-------------------|----------|
| **RTX 4090** | 185 tok/s | 完美，比云端慢一点但完全够用 |
| **RTX 3080/3090** | 120-150 tok/s | 很好 |
| **M2/M3 Max** | 50-80 tok/s | 可用，生成2500字约30-50秒 |
| **M2 Pro** | 30-50 tok/s | 可用，稍慢 |
| **M1 系列** | 20-40 tok/s | 能用，但建议用Groq云端 |

## 🆚 两个Agent对比

| Agent | 路径 | 引擎 | 优势 |
|-------|------|------|------|
| **Ministral-3 Local** | `agents/ministral-3-local/` | Ollama本地 | 免费、隐私、离线 |
| **Llama-3.3 Cloud** | `agents/llama33-70b/` | Groq云端 | 速度快、质量高 |

**都可以用，看你需求！**

## 📁 输出文件

```
three-books-ai-annotations/
└── YYYY-MM-DD-ai-annotation-local.md    # Ollama本地生成

books-ai-publishing/drafts/
└── YYYY-MM-DD-draft.md                  # 你的原始心得
```

## 🎯 每日工作流

### 方式1: 极简自动化（推荐）

```bash
./daily-4hour-life.sh
```

交互式菜单，自动完成：
1. 提醒你读书
2. 输入心得
3. 选择引擎（Ollama或Groq）
4. 自动注疏
5. 自动提交Git

### 方式2: 手动运行

```bash
# 写心得
vim today.txt

# 生成注疏
python3 agents/ministral-3-local/daily-annotation-ollama.py "$(cat today.txt)"

# 提交
git add . && git commit -m "daily" && git push
```

## 🔐 隐私说明

**Ollama本地运行特性**:
- ✅ 所有数据只在你的机器上
- ✅ 不发送任何数据到云端
- ✅ 完全离线可用
- ✅ 无需API key
- ✅ 无需担心配额限制

对于敏感内容（个人感悟、未发表想法），本地Ollama更安全。

## 🚀 高级用法

### 批量处理

```bash
# 如果你有多天积累的心得
for file in drafts/*.txt; do
    python3 agents/ministral-3-local/daily-annotation-ollama.py "$(cat $file)"
done
```

### 自定义Prompt

编辑 `daily-annotation-ollama.py` 第90行的prompt模板，调整输出风格。

## 🆘 故障排除

### 问题1: Ollama命令未找到

```bash
# 检查是否安装
which ollama

# 重新安装
curl -fsSL https://ollama.com/install.sh | sh
```

### 问题2: 模型未下载

```bash
# 手动下载
ollama pull ministral-3:8b

# 查看已下载模型
ollama list
```

### 问题3: 生成速度太慢

- **短期**: 切换到Groq云端 (Llama-3.3)
- **长期**: 考虑升级硬件（RTX 4090或M3 Max）

### 问题4: 中文乱码

```bash
export LANG=zh_CN.UTF-8
```

## 📈 推荐流程

**新手** → 先用Groq云端熟悉流程 → 再切换到Ollama本地
**老手** → 直接Ollama本地（免费+隐私）
**隐私敏感** → 只用Ollama本地
**追求速度** → Groq云端

## 🌟 2025-12-03 里程碑

今天Ministral-3发布是边缘AI的革命性时刻：

- **Apache 2.0开源**: 完全自由使用
- **8B参数**: 可在手机/树莓派上运行
- **性价比之王**: NVIDIA官方认证"best cost-performance for edge"
- **中文优秀**: 专门优化过中文处理

你的asicForTranAI项目正好赶上这波红利。

## 🎯 7年计划集成

Ollama本地 + Ministral-3 = 你的每日注疏引擎（2025-2032）

- **2025-2026**: 每天用Ollama生成注疏（完全免费）
- **2027**: 第一本书初稿完成
- **2028-2029**: 继续积累
- **2030-2032**: 4本书完成

**成本**: $0（Ollama完全免费）
**时间**: 每天1小时
**产出**: 7年后4本传世之作

---

**Created**: 2025-12-03
**Author**: Jim Xiao
**License**: MIT
**Model**: Ministral-3:8b @ Ollama
**完全本地，完全免费，完全开源**
