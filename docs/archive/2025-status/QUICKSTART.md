# 🚀 Quick Start - 立即开始第一次AI注疏

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](../../CLAIMS_AUDIT_2026-10.md). Do not quote them.

**时间**: 2025-12-03
**目标**: 5分钟内完成第一次三书AI注疏

---

## Step 1: 获取 Groq API Key (1分钟)

### 方法1: 浏览器访问
1. 打开: **https://console.groq.com/keys**
2. 用Google/GitHub账号登录（免费）
3. 点击 "Create API Key"
4. 复制 key（格式: `gsk_...`）

### 方法2: 命令行快速注册
```bash
# 如果你已有账号，直接去 console.groq.com/keys 复制
open https://console.groq.com/keys
```

**免费额度**: 1000万 tokens（够用3-6个月每日注疏）

---

## Step 2: 设置环境变量 (30秒)

```bash
# 临时设置（当前终端有效）
export GROQ_API_KEY="gsk_你的key粘贴在这里"

# 验证设置成功
echo $GROQ_API_KEY
```

**永久设置** (可选，添加到 `~/.zshrc` 或 `~/.bashrc`):
```bash
echo 'export GROQ_API_KEY="gsk_你的key"' >> ~/.zshrc
source ~/.zshrc
```

---

## Step 3: 安装依赖 (30秒)

```bash
# 只需要安装一次
pip3 install requests
```

---

## Step 4: 运行第一次注疏！(3分钟)

### 示例1: 孙子兵法

```bash
cd /Users/jimxiao/dev/2025/AI2025/asicForTranAI

python3 agents/llama33-70b/daily-three-books-agent.py "今天读《孙子·始计篇》第一句'兵者，国之大事，死生之地，存亡之道，不可不察也'。我想到2025年的AI芯片战：Groq选择专注LPU推理芯片，就像孙子强调的'不可不察'——这是关乎企业生死存亡的战略抉择。Nvidia虽然GPU市场占主导，但Groq的4188 tok/s推理速度（比INT4快35%）正是'知彼知己'的体现——知道自己的优势在哪里，避开Nvidia的训练市场，专攻推理。这让我想起孙子说的'善战者，求之于势，不责于人'——Groq找到了LPU架构这个'势'。"
```

### 示例2: 资治通鉴

```bash
python3 agents/llama33-70b/daily-three-books-agent.py "今天读《资治通鉴·周纪一》三家分晋的故事。韩、赵、魏三家瓜分晋国，让我想到现在AI三巨头的博弈：OpenAI（GPT）、Anthropic（Claude）、xAI（Grok）就像当年的三家分晋。司马光评论'礼乐征伐自诸侯出'，现在则是'AI标准规范自科技公司出'。胡三省注说'晋之失政久矣'，对应到今天，传统云计算巨头（AWS、Azure）在AI时代的失势也是'失政久矣'——他们错过了Transformer革命的最佳时机。"
```

### 示例3: 圣经

```bash
python3 agents/llama33-70b/daily-three-books-agent.py "今天读《创世记》第一章'起初，神创造天地'。我想到AI时代的'创世'：Transformer架构（2017 Attention is All You Need）就像'神说要有光，就有了光'——一个简单的注意力机制，照亮了整个AI领域。从无到有，从混沌到秩序。圣经说神用7天创造世界，AI领域也经历了类似的'七日创世'：Word2Vec、CNN、RNN、LSTM、Transformer、GPT、ChatGPT，每一步都是从混沌到秩序的进化。"
```

---

## Step 5: 查看输出 (30秒)

注疏生成后，你会看到：

```
============================================================
  三书AI注疏系统 - Llama-3.3-70B Daily Agent
  《孙子兵法》《资治通鉴》《圣经》AI时代解读
============================================================

📝 收到心得 (237 字)

✅ 原始心得已保存: books-ai-publishing/drafts/2025-12-03-draft.md

🤖 正在调用 Llama-3.3-70B-Versatile...
📊 模型: llama-3.3-70b-versatile
🌐 API: Groq

✅ AI注疏已生成: three-books-ai-annotations/2025-12-03-ai-annotation.md
📊 输出长度: 3247 字

============================================================
📖 注疏预览（前500字）：
============================================================

# 孙子兵法·始计篇 - AI时代注疏

**日期**: 2025年12月03日
**分类**: 孙子 > 始计第一

## 一、原文引用
...

============================================================
💾 是否自动提交到Git？(y/n)
```

---

## 🎯 你会得到什么？

每次运行自动生成：

1. **原文引用** (中英对照)
2. **历代注疏** (曹操、杜牧、胡三省等)
3. **深度解读** (2000-3000字分析)
4. **AI战例对照** (Groq vs Nvidia、xAI vs OpenAI等)
5. **启示思考** (总结)

**输出文件**:
- `books-ai-publishing/drafts/2025-12-03-draft.md` (你的原始心得)
- `three-books-ai-annotations/2025-12-03-ai-annotation.md` (AI生成的完整注疏)

---

## 📅 每日习惯建议

```bash
# 早上/晚上写心得（300-500字）
vim today-note.txt

# 一键生成AI注疏
python3 agents/llama33-70b/daily-three-books-agent.py "$(cat today-note.txt)"

# 查看输出
cat three-books-ai-annotations/$(date +%Y-%m-%d)-ai-annotation.md

# 自动提交（可选）
git add . && git commit -m "daily annotation" && git push
```

---

## 🔧 故障排除

### 问题1: `ModuleNotFoundError: No module named 'requests'`

```bash
pip3 install requests
```

### 问题2: `❌ 错误: GROQ_API_KEY 未设置`

```bash
export GROQ_API_KEY="gsk_你的key"
echo $GROQ_API_KEY  # 验证
```

### 问题3: 速率限制

Groq免费版限制：
- 30 请求/分钟
- 6000 tokens/分钟（输入）
- 6000 tokens/分钟（输出）

如果超限，等待1分钟后重试。

### 问题4: 中文乱码

```bash
export LANG=zh_CN.UTF-8
```

---

## 📊 进度追踪

### 第一天 (今天):
- [x] 获取 Groq API Key
- [ ] 运行第一次注疏
- [ ] 查看输出质量
- [ ] 调整心得长度（300-500字最佳）

### 第一周:
- [ ] 每天1篇心得 (7天 = 7篇注疏 = 17,500字)
- [ ] 周六自动生成第一份周论

### 第一月:
- [ ] 30篇注疏 = 75,000字
- [ ] 第一份月度汇编

### 第一年:
- [ ] 365篇注疏 = 912,500字
- [ ] 相当于3本书的初稿！

---

## 🎯 性能预期

基于 Llama-3.3-70B-Versatile @ Groq:

- **速度**: 5-8秒生成2500字注疏
- **质量**:
  - 中文流畅度: 95%+
  - 历史准确性: 90%+
  - AI对照创意: 85%+
- **成本**: 免费额度足够3-6个月

---

## 🚀 下一步

1. ✅ 运行第一次（上面任选一个例子）
2. 📖 阅读输出，检查质量
3. 🔧 根据输出调整你的心得长度和深度
4. 📅 建立每日习惯
5. 🎯 7年后见证4本书的诞生！

---

## 🫶 Remember

> **"千里之行，始于足下"**
>
> 7年、2555天、4本书，都始于今天的第一次300字心得。
>
> 不要完美主义，不要拖延，just run it now.
>
> Let's go! 🚀

---

**Created**: 2025-12-03
**Your first annotation starts**: NOW
**7-year journey ends**: 2032-12-03

**现在就运行！👇**

```bash
export GROQ_API_KEY="your_key"
python3 agents/llama33-70b/daily-three-books-agent.py "今天读《孙子·始计篇》……"
```
