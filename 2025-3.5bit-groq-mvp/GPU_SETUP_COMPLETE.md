# GPU Setup - Manual Steps (If Needed)

## Environment Created! ✓

**Python 3.9 environment "gpu" is ready**
- Location: `C:\Users\svden\.conda\envs\gpu`
- Python: 3.9.25
- NumPy: 2.0.2
- SciPy: 1.13.1

## PyTorch Installation (In Progress)

### Automatic (Currently Running)
The installation is running in the background. It may take 3-5 minutes for conda to solve dependencies.

### Manual (If You Want to Do It Yourself)

Open **Anaconda Prompt** and run:

```powershell
# Activate the gpu environment
conda activate gpu

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# This will download ~2-3 GB and take 5-10 minutes
```

**OR** use pip (faster alternative):

```powershell
# Activate the gpu environment
conda activate gpu

# Install via pip (usually faster)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Test GPU After Installation

```powershell
# Activate environment
conda activate gpu

# Test CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output:**
```
PyTorch: 2.x.x+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 2080 Ti
```

## Run Neural Network Tests

Once PyTorch is installed:

```powershell
# Activate environment
conda activate gpu

# Navigate to project
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp

# Run GPU tests
python test_gpu_neural_net.py

# Run full benchmarks
python benchmark_rtx2080ti.py
```

**Expected Performance:**
```
GPU: NVIDIA GeForce RTX 2080 Ti
Memory: 11.00 GB
CUDA: 11.8

BENCHMARK: PyTorch GPU
  1024x1024: 3.45 ms (6.12 TFLOPS)
  2048x2048: 25.67 ms (6.71 TFLOPS)
  4096x4096: 203.45 ms (6.75 TFLOPS)
```

## Troubleshooting

### Problem: "CUDA not available"

**Check 1**: GPU drivers
```powershell
nvidia-smi
# Should show RTX 2080 Ti
```

**Check 2**: PyTorch CUDA version
```powershell
conda activate gpu
python -c "import torch; print(torch.version.cuda)"
# Should show: 11.8
```

**Check 3**: Reinstall with correct CUDA version
```powershell
conda activate gpu
conda uninstall pytorch torchvision torchaudio -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Problem: "conda activate gpu" doesn't work

**Solution**: Initialize conda for your shell
```powershell
conda init powershell
# Close and reopen PowerShell/Anaconda Prompt
conda activate gpu
```

### Problem: Installation is slow

**Alternative**: Use pip instead of conda (usually 2-3x faster)
```powershell
# In Anaconda Prompt
conda activate gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start Commands

```powershell
# 1. Activate environment (do this every time you open a new terminal)
conda activate gpu

# 2. Check Python version
python --version
# Should show: Python 3.9.25

# 3. Test GPU
python -c "import torch; print(torch.cuda.is_available())"
# Should show: True

# 4. Run tests
cd C:\ai\asicForTranAI\2025-3.5bit-groq-mvp
python test_gpu_neural_net.py
```

## Status

- [x] Python 3.9 environment created
- [ ] PyTorch with CUDA installing (in progress)
- [ ] GPU tests pending
- [ ] Benchmarks pending

**Current install command running:**
```
conda install -n gpu pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

This is downloading and installing in the background. Check back in 5-10 minutes, or follow the manual steps above!

---

**Next Steps:**
1. Wait for automatic installation (or do manual install above)
2. Test GPU with: `python test_gpu_neural_net.py`
3. Run benchmarks: `python benchmark_rtx2080ti.py`
4. Start using your RTX 2080 Ti for neural network inference!
