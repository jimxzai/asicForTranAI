# 🚀 START HERE - Next 30 Minutes

**Goal**: Enable automation + Start career launch + Continue Fortran work
**Time**: 30 minutes to unlock everything

---

## ⚡ Action 1: MCP Setup (10 minutes)

**Copy-paste this into your terminal RIGHT NOW**:

```bash
# Install uv (for Git MCP)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell
source ~/.zshrc  # or source ~/.bashrc if you use bash

# Create MCP config directory
mkdir -p ~/.claude

# Create MCP settings
cat > ~/.claude/mcp_settings.json << 'MCPEOF'
{
  "mcpServers": {
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/Users/jimxiao/ai/asicForTranAI"]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/jimxiao/ai/asicForTranAI"
      ]
    }
  }
}
MCPEOF

echo ""
echo "✅ MCP configured!"
echo "⚠️  IMPORTANT: Restart Claude Code now"
echo "   1. Quit Claude Code completely"
echo "   2. Reopen it"
echo "   3. Come back to this chat"
```

**After restart, I'll have Git tools to auto-create your repos!**

---

## ⚡ Action 2: Test Groq API (10 minutes)

```bash
cd /Users/jimxiao/ai/asicForTranAI/2025-3.5bit-groq-mvp

# Check if API key is set
echo $GROQ_API_KEY

# If empty, set it:
# 1. Go to: https://console.groq.com
# 2. Sign up / login
# 3. Get API key
# 4. Run:
export GROQ_API_KEY="gsk_your_key_here"

# Test it
bash test_api_key.sh

# If working, you should see: ✅ API key is valid
```

---

## ⚡ Action 3: Read Your Career Materials (10 minutes)

```bash
cd /Users/jimxiao/ai/asicForTranAI/2025-3.5bit-groq-mvp/career

# Quick skim (2 minutes each):
cat LINKEDIN_PROFILE.md | head -100   # Your new LinkedIn headline
cat RESUME_TECHNICAL.md | head -100   # Resume template
cat ACTION_PLAN.md | head -100        # 16-week plan
```

**Note what needs filling in**: [BRACKETED] placeholders

---

## 📋 What You'll Have After 30 Minutes

✅ MCP automation ready (I can auto-create GitHub repos)
✅ Groq API working (ready to benchmark)
✅ Career materials reviewed (know what's next)

---

## 🎯 Tomorrow's Plan (After MCP is Working)

### Morning (2 hours):
1. I'll create GitHub repos using Git MCP (automated!)
2. You update LinkedIn (copy-paste from materials)
3. We run Groq benchmark together

### Afternoon (2 hours):
4. Polish resume (fill [BRACKETS])
5. Apply to AdaCore (first application!)
6. Network on LinkedIn (5 connections)

---

## 🆘 Troubleshooting

**"curl: command not found"**
- You're on Windows? Use WSL or download uv manually from https://astral.sh/uv/

**"source: no such file"**
- Try: `source ~/.bash_profile`
- Or just open a new terminal window

**"Groq API key not working"**
- Double-check you copied the whole key (starts with `gsk_`)
- Make sure no extra spaces
- Try in a new terminal window

**"Don't have Node.js for npx"**
- Install: `brew install node` (macOS)
- Or skip filesystem MCP for now (Git MCP is more important)

---

## 📞 What to Tell Me After Setup

Once you've run the commands above and restarted Claude Code, just say:

**"MCP ready"** or **"Setup done"**

Then I'll:
1. ✅ Verify Git MCP is working
2. ✅ Create spark-llama-safety repo (automated)
3. ✅ Push all your career materials
4. ✅ Help you with Groq benchmark

---

## 🎁 Bonus: Full File Map (What You Have)

```
/Users/jimxiao/ai/asicForTranAI/
├── START_HERE.md                          ← THIS FILE
├── MASTER_PLAN_ABC.md                     ← Full A+B+C integration
├── CAREER_PACKAGE_READY.md                ← Career launch guide
│
├── 2025-3.5bit-groq-mvp/                  ← Fortran project
│   ├── matmul_3p5bit_dynamic.f90          ← Fixed code (ready!)
│   ├── CODE_REVIEW_SUMMARY.md             ← Technical review
│   ├── career/
│   │   ├── LINKEDIN_PROFILE.md            ← LinkedIn optimization
│   │   ├── RESUME_TECHNICAL.md            ← Resume template
│   │   ├── COVER_LETTERS.md               ← 3 cover letters
│   │   └── ACTION_PLAN.md                 ← 16-week roadmap
│   └── ... (tests, benchmarks, etc.)
│
└── spark-llama-safety/                    ← Portfolio project
    ├── README.md                          ← "World first" claim
    ├── src/quantization.ads               ← Ada code sample
    └── Makefile                           ← Proof workflow
```

---

**Time check**: If you're reading this, you're 5 minutes in. Spend 10 more on MCP setup, 10 on Groq, 5 on reading. Then you're done for today!

**Tomorrow**: We execute. GitHub, LinkedIn, applications, benchmarks. ALL OF IT.

**Let's go! 🚀**
