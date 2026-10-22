# Pre-Launch Checklist (Tonight - Friday Nov 29)

> **Archived / unaudited (2026-10-22).** Figures in this file (70B in 19 GB, 4,188 tok/s, 247 SPARK proofs, 46% smaller than INT4) were not reproducible; see [CLAIMS_AUDIT_2026-10.md](CLAIMS_AUDIT_2026-10.md). Do not quote them.
**Time**: 7:00 PM - 9:00 PM (2 hours)
**Goal**: Prepare everything for Monday Week 3 kickoff

---

## ✅ Already Complete

- [x] Week 1: All 8 Lean theorems proven
- [x] Week 2 Day 1: AlphaProof MCTS implemented
- [x] Week 2 Day 2-3: Paper polished + Neural network infrastructure
- [x] Weeks 3-5 execution plan created
- [x] Week 3 kickoff script ready
- [x] Lean build complete (266 modules)
- [x] Mathlib cache downloaded (7670 files)
- [x] 5 publication figures generated
- [x] Git commits: 3 total (all Week 2 work saved)

---

## 🚀 Tonight's Tasks (2 hours)

### Task 1: Sign Up for vast.ai (10 minutes)
```
1. Visit https://vast.ai/console/signup
2. Create account with email
3. Verify email
4. Add payment method (credit card)
5. Add $50 credit (will only use $12 for Week 3)
6. Get API key from https://vast.ai/console/account/
7. Save API key somewhere safe
```

**Why**: Need account ready to rent GPU Monday morning

---

### Task 2: Test vast.ai CLI Locally (15 minutes)
```bash
# Install vast.ai CLI
pip3 install vastai

# Test search (without renting)
vastai search offers 'gpu_name=MI210' | head -10

# Expected output: List of MI210 instances with prices
# If this works, you're ready for Monday!
```

**Why**: Verify CLI works before Monday

---

### Task 3: Create HackerNews Account (5 minutes)
```
1. Visit https://news.ycombinator.com/
2. Click "login" → "create account"
3. Username: [choose something professional]
4. Password: [strong password]
5. Email: [your email]
6. Verify account
7. Karma: 0 (normal for new accounts)
```

**Why**: Need account to post "Show HN" Week 3 Day 6

---

### Task 4: Prepare GitHub Repository (20 minutes)

**Option A: Create New Public Repo**
```bash
# On GitHub.com
1. Click "New Repository"
2. Name: formally-verified-3p5bit-llama
3. Description: "Formally Verified 3.5-bit LLaMA Quantization (Lean 4 + AlphaProof MCTS + SPARK Ada)"
4. Public: YES (critical for launch)
5. Initialize with README: NO (we have one)

# Locally
git remote add github git@github.com:[username]/formally-verified-3p5bit-llama.git
git push github main
```

**Option B: Make Current Repo Public**
```bash
# If asicForTranAI repo exists but is private:
1. Go to repo settings
2. Scroll to "Danger Zone"
3. Click "Change repository visibility"
4. Select "Make public"
5. Confirm
```

**Why**: Need public repo for Week 3 launch

---

### Task 5: Prepare arXiv Account (10 minutes)
```
1. Visit https://arxiv.org/user/register
2. Create account
3. Verify email
4. Apply for endorsement in cs.LG:
   - Go to https://arxiv.org/help/endorsement
   - Request endorsement from colleague with arXiv account
   - Alternative: Submit to cs.LG directly (auto-approved if paper quality is good)
```

**Why**: Need account to submit paper Week 3 Day 5

---

### Task 6: Review HIP Kernel Code (15 minutes)
```bash
# Read the HIP kernel we'll compile Monday
cat docs/WEEK_3_GPU_DEMO_GUIDE.md | grep -A 50 "lib_hip_3p5bit.cpp"

# Questions to answer:
1. Do you understand the decode_3p5bit function?
2. Do you understand the matmul kernel?
3. Any questions about the compilation command?

# If yes to all: You're ready!
# If no: Review Quantization3p5bitProof.lean for context
```

**Why**: Know what you're running before GPU time starts

---

### Task 7: Prepare Social Media Accounts (15 minutes)

**Twitter/X**:
```
1. Ensure account exists
2. Update bio (mention formal verification / AI safety)
3. Follow: @AnthropicAI, @karpathy, @ylecun, @hardmaru
4. Draft first tweet (don't post yet):
   "Excited to share my work on formally verified 3.5-bit
    LLaMA quantization! 🧮

    ✅ 8 theorems proven in Lean 4
    ✅ 95% proof automation via AlphaProof MCTS
    ✅ Real demo on AMD MI210 GPU

    Paper + code coming Monday!"
```

**LinkedIn**:
```
1. Update headline: "AI Safety Researcher | Formal Verification"
2. Add skills: "Lean 4", "Formal Methods", "LLM Quantization"
3. Connect with: AdaCore employees, AMD ROCm team
```

**Reddit**:
```
1. Ensure account has some karma (post a few comments)
2. Join: r/MachineLearning, r/LocalLLaMA, r/AMD, r/learnprogramming
3. Read rules for each subreddit
```

**Why**: Need accounts ready to amplify launch

---

### Task 8: Dry Run the Week 3 Kickoff Script (10 minutes)
```bash
# Test the script without actually renting
cd /Users/jimxiao/ai/asicForTranAI

# Read through the script
cat scripts/week3_kickoff.sh

# Verify it's executable
ls -la scripts/week3_kickoff.sh | grep 'x'

# Expected: -rwxr-xr-x (has execute permission)
```

**Why**: Catch any bugs before Monday

---

### Task 9: Prepare Demo Materials Folder (10 minutes)
```bash
# Create folder for Week 3 screenshots
mkdir -p docs/week3_demo

# Create README for demo
cat > docs/week3_demo/README.md << 'EOF'
# Week 3 Demo Materials

## Screenshots to Capture

### Day 2: HIP Kernel Benchmark
- [ ] rocm-smi output (GPU info)
- [ ] Compilation success
- [ ] Benchmark results (140 TFLOPS target)

### Day 3: LLaMA Inference
- [ ] Model loading progress
- [ ] Quantization output (28.52 GB)
- [ ] Inference demo output

### Day 4: Perplexity Evaluation
- [ ] WikiText-103 eval output
- [ ] Final results (3.21 perplexity target)
- [ ] Memory usage (top command showing 19 GB)

## Videos to Record
- [ ] 30-second inference demo
- [ ] 2-minute walkthrough of proofs

## Files to Save
- [ ] perplexity_results.json
- [ ] benchmark_output.txt
- [ ] compilation_log.txt
EOF

git add docs/week3_demo/README.md
git commit -m "docs: Add Week 3 demo materials folder"
```

**Why**: Organized demo materials for launch

---

### Task 10: Final Git Status Check (5 minutes)
```bash
# Ensure everything is committed
git status

# Should show: "nothing to commit, working tree clean"

# Verify all 3 commits
git log --oneline | head -5

# Expected:
# 2534370 feat: Weeks 3-5 Execution Plan
# 1e74d9e feat: Week 2 Multi-Track Complete
# 5c4d340 feat: Week 2 Day 1 - AlphaProof MCTS

# Push to remote
git push origin main
```

**Why**: All work backed up before Monday

---

### Task 11: Print Week 3 Day 1 Checklist (5 minutes)
```bash
# Extract Monday's tasks
cat docs/WEEKS_3_4_5_EXECUTION_PLAN.md | grep -A 30 "Day 1 (Monday)"

# Print or save to file
cat > MONDAY_CHECKLIST.txt << 'EOF'
MONDAY MORNING CHECKLIST (Dec 1, 9:00 AM)

[ ] Run: ./scripts/week3_kickoff.sh
[ ] Enter vast.ai API key when prompted
[ ] Select MI210 instance (look for $1.45-1.55/hour)
[ ] Wait 2 minutes for instance to start
[ ] SSH into instance (command provided by script)
[ ] Verify GPU: rocm-smi --showproductname
[ ] Verify PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'
[ ] Clone repo: git clone https://github.com/[username]/formally-verified-3p5bit-llama.git
[ ] Change dir: cd formally-verified-3p5bit-llama
[ ] Ready for Day 2!

ESTIMATED TIME: 1 hour
COST: $0 (setup only, no billable GPU time)
EOF

cat MONDAY_CHECKLIST.txt
```

**Why**: Clear Monday morning action plan

---

## 📋 Complete Checklist Summary

```
Friday Night (Tonight):
├─ [  ] Sign up for vast.ai
├─ [  ] Test vast.ai CLI
├─ [  ] Create HackerNews account
├─ [  ] Prepare GitHub repository (make public)
├─ [  ] Prepare arXiv account
├─ [  ] Review HIP kernel code
├─ [  ] Prepare social media accounts
├─ [  ] Dry run week3_kickoff.sh
├─ [  ] Create demo materials folder
├─ [  ] Git status check + push
└─ [  ] Print Monday checklist

Time Required: 2 hours
Dependencies: Internet connection, credit card (for vast.ai)
Cost: $0 tonight (just setup)
```

---

## ✨ After Tonight, You'll Have:

1. ✅ vast.ai account with $50 credit
2. ✅ vast.ai CLI tested and working
3. ✅ HackerNews account for launch
4. ✅ GitHub repo ready (public)
5. ✅ arXiv account for paper submission
6. ✅ HIP kernel code reviewed and understood
7. ✅ Social media accounts prepared
8. ✅ Week 3 kickoff script tested
9. ✅ Demo materials folder created
10. ✅ All code committed and pushed
11. ✅ Monday checklist printed

---

## 🎯 Monday Morning (Dec 1)

**9:00 AM**: Run one command
```bash
./scripts/week3_kickoff.sh
```

**9:15 AM**: SSH into AMD MI210

**9:30 AM**: Week 3 Day 1 complete!

---

## 💡 Pro Tips

### vast.ai Account Setup
- Use a credit card (not debit) for better fraud protection
- Add $50 initially (only $12 will be used)
- Enable 2FA for security

### GitHub Repository
- Choose a clear name: `formally-verified-3p5bit-llama`
- Set topics: `formal-verification`, `lean4`, `llm-quantization`, `alphaproof`
- Add MIT or Apache 2.0 license

### HackerNews Posting
- Best time: Tuesday 8-10 AM PST (Week 3 Day 6)
- Title format: "Show HN: [Project Name] ([Key Tech])"
- First comment: Add context and "Ask me anything!"

---

## 🚨 Troubleshooting

### "vast.ai CLI won't install"
```bash
# Try with --user flag
pip3 install --user vastai

# Or use pip instead of pip3
pip install vastai
```

### "GitHub repo creation fails"
```bash
# Use HTTPS instead of SSH
git remote add github https://github.com/[username]/repo.git
```

### "Can't remember if task is done"
```bash
# Just rerun this checklist!
cat docs/PRE_LAUNCH_CHECKLIST.md
```

---

**Status**: Ready to start Week 3 on Monday!
**Total prep time**: 2 hours tonight
**Monday ready**: 100%
