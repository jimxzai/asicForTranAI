# LLaMA 70B Scripts

Helper scripts for weight loading, tokenization, and MLIR generation.

---

## 📦 Prerequisites

```bash
# Install Python dependencies
pip install huggingface_hub safetensors numpy sentencepiece
```

---

## 🔧 Weight Management

### 1. Download LLaMA Weights (~140GB)

```bash
./scripts/download_llama_weights.py

# Or with custom output directory
./scripts/download_llama_weights.py --output /path/to/weights
```

**What it does:**
- Downloads LLaMA 70B AWQ weights from Hugging Face
- Saves ~140GB of safetensors files
- Resumes automatically if interrupted

---

### 2. Convert to Fortran Binary

```bash
./scripts/convert_weights_to_fortran.py

# Or with custom paths
./scripts/convert_weights_to_fortran.py \
  --input ./llama-70b-awq \
  --output ./weights_fortran \
  --layers 80
```

**What it does:**
- Converts safetensors → Fortran binary format
- Creates one `.bin` file per layer (80 total)
- Generates metadata JSON files
- Each layer ~100MB

**Output:**
```
weights_fortran/
├── weights_layer0.bin
├── weights_layer0_meta.json
├── weights_layer1.bin
├── weights_layer1_meta.json
...
└── weights_layer79_meta.json
```

---

## 📝 Tokenization

### Setup

Download the LLaMA tokenizer:

```bash
# Option 1: From Hugging Face (requires authentication)
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-70b-hf tokenizer.model

# Option 2: Use TheBloke's version (public)
wget https://huggingface.co/TheBloke/Llama-2-70B-AWQ/raw/main/tokenizer.model
```

### Usage

**Encode text → token IDs:**

```bash
./scripts/tokenizer.py encode "Hello, world!"

# Output:
# Text: Hello, world!
# Tokens: 4
# Token IDs: [1, 15043, 29892, 3186, 29991]

# Save to binary file (for Fortran)
./scripts/tokenizer.py encode "Hello, world!" --output prompt.bin
```

**Decode token IDs → text:**

```bash
./scripts/tokenizer.py decode prompt.bin

# Output:
# Decoded text:
# Hello, world!
```

**Interactive mode:**

```bash
./scripts/tokenizer.py interactive

# Enter text: The quick brown fox
#   Tokens: 5
#   Token IDs: [1, 450, 4996, 17354, 1701, 29916]
#   Decoded: The quick brown fox
```

---

## 🏗️ MLIR Generation

Generate MLIR from Fortran source (for ASIC deployment):

```bash
./scripts/generate_mlir.sh
```

**What it does:**
- Uses LFortran to generate MLIR
- Creates `llama_model.mlir`
- Ready for Groq compiler

---

## 🧪 Lean4 Formal Verification

Set up Lean4 for formal proofs:

```bash
./scripts/setup_lean4.sh
```

**What it does:**
- Installs Lean4 toolchain
- Sets up quantization proofs
- Creates project structure

---

## 🔄 Complete Workflow

### Step-by-Step Guide

```bash
# 1. Install dependencies
pip install huggingface_hub safetensors numpy sentencepiece

# 2. Download weights (~1-4 hours depending on connection)
./scripts/download_llama_weights.py

# 3. Convert to Fortran binary (~30 minutes)
./scripts/convert_weights_to_fortran.py

# 4. Download tokenizer
wget https://huggingface.co/TheBloke/Llama-2-70B-AWQ/raw/main/tokenizer.model

# 5. Generate test weights (if you don't want to download 140GB)
cd ..
make gen-weights

# 6. Test tokenizer
./scripts/tokenizer.py encode "Once upon a time" --output prompt.bin

# 7. Build and run inference
make test-model

# 8. (Future) Generate MLIR for ASIC
./scripts/generate_mlir.sh
```

---

## 📊 File Sizes

| Item | Size | Description |
|------|------|-------------|
| Downloaded weights | ~140 GB | Original safetensors |
| Converted weights | ~8 GB | Fortran binary (80 layers × 100MB) |
| Tokenizer model | ~500 KB | SentencePiece model |
| Test weights | ~100 MB | Random weights for testing |

---

## 🐛 Troubleshooting

### "huggingface_hub not found"
```bash
pip install --upgrade huggingface_hub
```

### "safetensors not found"
```bash
pip install safetensors
```

### "sentencepiece not found"
```bash
pip install sentencepiece
```

### "Permission denied"
```bash
chmod +x scripts/*.py scripts/*.sh
```

### "Tokenizer model not found"
Download from: https://huggingface.co/TheBloke/Llama-2-70B-AWQ/raw/main/tokenizer.model

---

## 📝 Notes

- **Disk Space**: Ensure you have ~150GB free for weights
- **Download Time**: Weights download can take 1-4 hours
- **Conversion Time**: Weight conversion takes ~30 minutes
- **Memory**: Weight loading requires ~32GB RAM minimum

---

## 🔗 Links

- **LLaMA Weights**: https://huggingface.co/TheBloke/Llama-2-70B-AWQ
- **Tokenizer**: https://huggingface.co/meta-llama/Llama-2-70b-hf
- **Hugging Face Docs**: https://huggingface.co/docs/hub/
- **SentencePiece**: https://github.com/google/sentencepiece

---

## ✅ Quick Test (No Downloads Required)

Test the pipeline without downloading weights:

```bash
# 1. Generate test weights
make gen-weights

# 2. This creates test_weights_layer0.bin (~100MB)

# 3. Run tests
make test-model

# 4. Verify everything compiles and runs

# 5. Later, replace with real weights when ready
```

This way you can develop and test without waiting for 140GB downloads!
