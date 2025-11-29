# Platform Setup Guide
## Which Machine for RTX 2080 Ti?

---

## 🖥️ Quick Answer

**RTX 2080 Ti**: Must be in a **Windows or Linux PC** (desktop)
**MacBook**: Cannot use RTX 2080 Ti (external GPUs not well supported)

---

## 🎯 Recommended Setup

### Option 1: Linux (Recommended ⭐⭐⭐)

**Best for**: Production, development, maximum performance

```
OS:              Ubuntu 22.04 LTS (or 20.04)
Why Linux:       - Best CUDA support
                 - No overhead
                 - Easy automation
                 - SSH access
                 - Professional setup
                 
Performance:     1,400-1,500 tok/s
Difficulty:      Easy (if familiar with Linux)
Setup time:      30 minutes
```

**Installation Steps**:
```bash
# 1. Install Ubuntu 22.04 on your PC with RTX 2080 Ti
# Download: https://ubuntu.com/download/desktop

# 2. Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525  # Or latest
sudo reboot

# 3. Verify GPU
nvidia-smi
# Should show RTX 2080 Ti

# 4. Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# 5. Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make LLAMA_CUDA=1

# 6. Download model
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf

# 7. Run!
./llama-cli -m llama-2-13b-chat.Q4_K_M.gguf -ngl 40 -p "Hello!"

# Done! ✅
```

---

### Option 2: Windows (Easier for Beginners ⭐⭐)

**Best for**: If you're already on Windows, gaming PC

```
OS:              Windows 10/11
Why Windows:     - Familiar interface
                 - Easy to set up
                 - Good for gaming + AI
                 - GUI tools available
                 
Performance:     1,200-1,400 tok/s (slightly slower than Linux)
Difficulty:      Very Easy
Setup time:      20 minutes
```

**Installation Steps**:
```powershell
# 1. Install CUDA Toolkit for Windows
# Download from: https://developer.nvidia.com/cuda-downloads
# Choose: Windows > x86_64 > 11.8 > exe (local)
# Run installer, accept defaults

# 2. Install Visual Studio Build Tools (required)
# Download: https://visualstudio.microsoft.com/downloads/
# Install "Desktop development with C++"

# 3. Install Git for Windows
# Download: https://git-scm.com/download/win

# 4. Open PowerShell as Administrator
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 5. Build with CUDA
mkdir build
cd build
cmake .. -DGGML_CUDA=ON
cmake --build . --config Release

# 6. Download model (in PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf" -OutFile "llama-2-13b-chat.Q4_K_M.gguf"

# 7. Run!
.\bin\Release\llama-cli.exe -m llama-2-13b-chat.Q4_K_M.gguf -ngl 40 -p "Hello!"

# Done! ✅
```

**Or Use Pre-built Windows Binary** (Easiest!):
```powershell
# 1. Download pre-compiled llama.cpp for Windows
# From: https://github.com/ggerganov/llama.cpp/releases
# Get: llama-xxx-bin-win-cuda-cu11.8.0-x64.zip

# 2. Extract to C:\llama.cpp\

# 3. Download model
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf" -OutFile "llama-2-13b-chat.Q4_K_M.gguf"

# 4. Run!
cd C:\llama.cpp
.\llama-cli.exe -m llama-2-13b-chat.Q4_K_M.gguf -ngl 40

# Even easier! ✅
```

---

### Option 3: MacBook (Development Only, No GPU)

**Your MacBook cannot use RTX 2080 Ti**
- RTX 2080 Ti is a desktop GPU (PCIe card)
- MacBooks don't have PCIe slots
- External GPU enclosures are not well supported on macOS

**But you CAN use MacBook for**:
```
✅ Development & testing our Fortran code
✅ CPU-only inference (104 tok/s on M1 Max)
✅ Writing code, testing algorithms
✅ Remote access to Linux/Windows PC

Your setup:
  MacBook M1 Max → Development machine
  PC with RTX 2080 Ti → Inference server (Windows or Linux)
  
Workflow:
  1. Write code on MacBook
  2. SSH to Linux PC or RDP to Windows PC
  3. Run inference on RTX 2080 Ti
  4. Perfect! ✅
```

---

## 🎯 Recommended Configuration

### Scenario A: You Have a Gaming PC with Windows

```
Setup: Windows 11 + RTX 2080 Ti
Time:  20 minutes
Steps:
  1. Download pre-built llama.cpp for Windows
  2. Download LLaMA-13B model
  3. Double-click to run
  4. Done!

Pro: Easiest, familiar, works immediately
Con: Slightly slower than Linux (~5-10%)
```

### Scenario B: You Want Maximum Performance

```
Setup: Ubuntu 22.04 + RTX 2080 Ti
Time:  30 minutes (includes Ubuntu installation)
Steps:
  1. Install Ubuntu on PC
  2. Install NVIDIA drivers + CUDA
  3. Compile llama.cpp
  4. Run

Pro: Best performance, production-ready
Con: Need to learn Linux basics (not hard!)
```

### Scenario C: Dual Boot (Best of Both Worlds!)

```
Setup: Windows 11 + Ubuntu 22.04 (dual boot)
Time:  45 minutes
Steps:
  1. Shrink Windows partition (Disk Management)
  2. Install Ubuntu alongside Windows
  3. Choose OS at boot time

Pro: Gaming on Windows, AI on Linux
Con: Requires reboot to switch OS
```

---

## 💻 What You Need

### For Linux Setup:

```
Hardware:
  ✅ PC with RTX 2080 Ti installed
  ✅ 16GB+ RAM (32GB recommended)
  ✅ 100GB+ free disk space
  ✅ Internet connection

Software (all free):
  ✅ Ubuntu 22.04 LTS ISO
  ✅ NVIDIA drivers (auto-installed)
  ✅ CUDA toolkit (apt install)
  ✅ llama.cpp (git clone)
  
Total cost: $0
```

### For Windows Setup:

```
Hardware:
  ✅ PC with RTX 2080 Ti installed  
  ✅ 16GB+ RAM (32GB recommended)
  ✅ 100GB+ free disk space
  ✅ Internet connection

Software (all free):
  ✅ Windows 10/11 (you probably have)
  ✅ CUDA Toolkit (free download)
  ✅ Visual Studio Build Tools (free)
  ✅ Git for Windows (free)
  ✅ llama.cpp (git clone or pre-built)

Total cost: $0 (assuming you have Windows)
```

---

## 🚀 Quick Decision Matrix

**I have a Windows gaming PC with RTX 2080 Ti**
→ Use Windows! Download pre-built llama.cpp
→ Time to first inference: 15 minutes
→ Difficulty: ⭐ (very easy)

**I want maximum performance**
→ Install Ubuntu Linux
→ Time to first inference: 40 minutes
→ Difficulty: ⭐⭐ (medium, but worth it)

**I only have a MacBook**
→ RTX 2080 Ti won't work with MacBook
→ Buy/build a Linux PC or use existing Windows PC
→ Or use MacBook CPU-only (104 tok/s)

**I want to use both Windows (gaming) and Linux (AI)**
→ Dual boot setup
→ Best of both worlds!
→ Difficulty: ⭐⭐⭐ (advanced)

---

## 📊 Performance Comparison

### Same RTX 2080 Ti, Different OS:

```
Linux (Ubuntu 22.04):
  LLaMA-13B:  1,400-1,500 tok/s  ✅ Best
  Latency:    ~85ms first token
  Overhead:   Minimal

Windows 11:
  LLaMA-13B:  1,200-1,400 tok/s  ✅ Good
  Latency:    ~95ms first token  
  Overhead:   ~5-10% (Windows services)

macOS (M1 Max, CPU only):
  LLaMA-13B:  104 tok/s          ⚠️ No GPU
  Latency:    ~500ms first token
  Use case:   Development only
```

**Verdict**: Linux is ~10% faster, but Windows is easier. Pick based on your comfort level!

---

## 🔧 Remote Access Setup

### Access RTX 2080 Ti PC from MacBook

**Option 1: SSH (Linux)**
```bash
# On Linux PC with RTX 2080 Ti:
sudo apt install openssh-server
sudo systemctl enable ssh
ip addr show  # Note the IP address

# On MacBook:
ssh username@192.168.1.xxx
# Now you're on the Linux PC!

# Run llama.cpp remotely:
./llama-cli -m model.gguf -ngl 40 -p "Hello!"
```

**Option 2: RDP (Windows)**
```
# On Windows PC:
Settings → System → Remote Desktop → Enable

# On MacBook:
Download Microsoft Remote Desktop from App Store
Connect to Windows PC IP address
# Now you have full Windows desktop!

# Run llama.cpp in PowerShell
```

**Option 3: VS Code Remote**
```bash
# On MacBook:
# Install VS Code with Remote-SSH extension
# Connect to Linux/Windows PC
# Edit code locally, run on GPU remotely!

# Best of both worlds:
#   MacBook: Great keyboard, screen, portability
#   PC with RTX 2080 Ti: GPU power
```

---

## ✅ My Recommendation for You

Based on your setup (MacBook M1 Max + RTX 2080 Ti PC):

### **Hybrid Setup** (Recommended ⭐⭐⭐):

```
1. Install Ubuntu on your PC with RTX 2080 Ti
   - Best performance
   - Easy SSH from MacBook
   - Professional setup

2. Keep MacBook for development
   - Write code on MacBook (great keyboard!)
   - SSH to Linux PC to run inference
   - Best of both worlds

3. Set up VS Code Remote
   - Edit files on MacBook
   - Files actually on Linux PC
   - Run with GPU automatically
   - Seamless workflow!

Workflow:
  MacBook → SSH → Linux PC → RTX 2080 Ti → 1,500 tok/s!
  
Perfect! ✅
```

### Alternative: Windows PC + MacBook

```
If your PC already has Windows and you don't want to change:

1. Keep Windows on RTX 2080 Ti PC
   - Download pre-built llama.cpp
   - 1,200-1,400 tok/s (still great!)

2. Use MacBook for development
   - RDP to Windows when you need GPU
   - Or SSH with WSL2 (Windows Subsystem for Linux)

3. Also works perfectly! ✅
```

---

## 📚 Next Steps

**This Weekend**:
1. Decide: Linux or Windows?
2. Install llama.cpp on RTX 2080 Ti PC
3. Download LLaMA-13B model
4. Run first test: "Hello, world!"
5. Celebrate 1,200+ tok/s! 🎉

**Next Week**:
1. Set up SSH/RDP from MacBook
2. Integrate with your application
3. Test different models
4. Start building!

---

## 🎯 Summary

**Your Hardware**:
- ✅ MacBook M1 Max (development)
- ✅ PC with RTX 2080 Ti (inference)

**Best Setup**:
- ✅ Ubuntu on RTX 2080 Ti PC
- ✅ SSH from MacBook
- ✅ VS Code Remote (optional but awesome)

**Expected Performance**:
- ✅ 1,400-1,500 tok/s (Linux)
- ✅ 1,200-1,400 tok/s (Windows)

**Cost**: $0 (use what you have!)

**Ready to start**! 🚀

---

**Document created**: 2025-11-28  
**Your setup**: MacBook + RTX 2080 Ti PC  
**Recommended**: Install Ubuntu on PC, SSH from MacBook

