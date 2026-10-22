# 🔧 Installing GNAT Studio + SPARK (macOS)

## Quick Summary
GNAT Studio (Ada IDE) + GNATprove (SPARK verifier) are **free** but require manual download from AdaCore (not available via brew).

---

## Option 1: GNAT Community Edition (Recommended)

### Step 1: Download
Visit: **https://www.adacore.com/download**

Select:
- **Product**: GNAT Studio
- **Platform**: macOS (x86_64 or ARM64, depending on your Mac)
- **Version**: 2024 (latest stable)
- **Package**: Community Edition (includes SPARK/GNATprove)

**File size**: ~500 MB
**Registration**: Free (email required)

### Step 2: Install
```bash
# Download completes to ~/Downloads/gnat-*.dmg
open ~/Downloads/gnat-2024-*-darwin-bin.dmg

# Drag GNAT Studio to /Applications
# Or use installer script:
sudo /Volumes/GNAT\ 2024/doinstall
```

**Installation path**: `/opt/GNAT/2024/` (default)

### Step 3: Add to PATH
```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="/opt/GNAT/2024/bin:$PATH"

# Reload shell
source ~/.zshrc
```

### Step 4: Verify
```bash
# Check GNAT compiler
gnat --version
# Expected: GNAT Community 2024 (20240523-103)

# Check GNATprove (SPARK verifier)
gnatprove --version
# Expected: GNATprove 14.1.0

# Check GPS (IDE)
gnatstudio --version
# Expected: GNAT Studio 24.0
```

---

## Option 2: GNAT FSF via Homebrew (Fallback)

**Note**: This is the **GNU Ada compiler** (not AdaCore's version), missing some SPARK features. Use only if AdaCore download fails.

```bash
brew install gcc
brew install gnat

# Verify
gnat --version
# Expected: GNAT FSF 13.x or 14.x
```

**Limitation**: No GNATprove (SPARK verifier) included. You'll need to build from source:
```bash
git clone https://github.com/AdaCore/spark2014.git
cd spark2014
make setup
make install
```

⚠️ **Building SPARK from source takes 1-2 hours**. Use Option 1 unless absolutely necessary.

---

## Option 3: Docker (Quickest for Testing)

AdaCore provides official Docker images:

```bash
# Pull latest GNAT Community
docker pull adacore/gnat-ce:latest

# Run interactive shell
docker run -it -v $(pwd):/workspace adacore/gnat-ce bash

# Inside container:
cd /workspace/spark-llama-safety
gnatprove -P transformer.gpr --level=2
```

**Pros**: No installation, instant access
**Cons**: Slower I/O, no GUI (GNATstudio won't work)

---

## Post-Install: Test SPARK Verification

### 1. Create Test Project
```bash
cd /tmp
mkdir spark_test && cd spark_test

# Create simple Ada package
cat > hello.ads <<'EOF'
package Hello with SPARK_Mode => On is
   function Add (X, Y : Integer) return Integer
     with Pre  => X >= 0 and Y >= 0,
          Post => Add'Result = X + Y;
end Hello;
EOF

cat > hello.adb <<'EOF'
package body Hello with SPARK_Mode => On is
   function Add (X, Y : Integer) return Integer is
   begin
      return X + Y;
   end Add;
end Hello;
EOF

# Create project file
cat > hello.gpr <<'EOF'
project Hello is
   for Source_Dirs use (".");
   for Object_Dir use "obj";
   package Prove is
      for Proof_Switches ("Ada") use ("--level=2");
   end Prove;
end Hello;
EOF
```

### 2. Run GNATprove
```bash
gnatprove -P hello.gpr

# Expected output:
# Phase 1 of 2: generation of Global contracts ...
# Phase 2 of 2: flow analysis and proof ...
# hello.adb:4:14: info: postcondition proved
# hello.adb:4:14: info: range check proved
# Summary: 2 checks proved, 0 checks not proved
```

**Success**: If you see `2 checks proved`, SPARK is working! ✅

---

## Troubleshooting

### Issue 1: "gnatprove: command not found"
**Cause**: PATH not set correctly
**Fix**:
```bash
# Find GNAT installation
find /opt -name gnatprove 2>/dev/null
# Or: find /usr/local -name gnatprove

# Add to PATH (use found path)
export PATH="/opt/GNAT/2024/bin:$PATH"
```

### Issue 2: "cannot find why3"
**Cause**: SPARK backend missing (rare with AdaCore installer)
**Fix**: Re-run AdaCore installer, ensure "SPARK" checkbox selected

### Issue 3: macOS Gatekeeper blocks execution
**Cause**: Unsigned binary (security feature)
**Fix**:
```bash
# Allow GNAT binaries
xattr -cr /opt/GNAT/2024/

# Or: System Preferences → Security & Privacy → Allow anyway
```

### Issue 4: Proof fails with timeout
**Cause**: Complex contracts, slow SMT solver
**Fix**:
```bash
# Increase timeout (default 1s → 60s)
gnatprove -P project.gpr --timeout=60

# Use faster solver (CVC5 instead of Z3)
gnatprove -P project.gpr --prover=cvc5
```

---

## Next Steps: Verify Your Transformer Layer

Once GNAT is installed:

```bash
cd /Users/jimxiao/ai/asicForTranAI/spark-llama-safety

# Create project file (if not exists)
cat > transformer.gpr <<'EOF'
project Transformer is
   for Source_Dirs use (".");
   for Object_Dir use "obj";
   package Prove is
      for Proof_Switches ("Ada") use ("--level=4", "--timeout=60");
   end Prove;
end Transformer;
EOF

# Run verification (start with level 2 for speed)
gnatprove -P transformer.gpr --level=2

# Expected: ~300 proof obligations
# Target: 95%+ proven automatically
```

---

## Resources

- **AdaCore Learn**: https://learn.adacore.com/courses/intro-to-spark/
- **SPARK User Guide**: https://docs.adacore.com/spark2014-docs/html/ug/
- **GNATprove Manual**: https://docs.adacore.com/gnatprove-docs/html/gnatprove_ug.html
- **Community Forum**: https://forum.ada-lang.io/

---

## Installation Status Checklist

After following steps above, check:

- [ ] `gnat --version` works
- [ ] `gnatprove --version` works
- [ ] Test project proves 2/2 checks
- [ ] Ready to verify transformer layer

**Once all checked, you're ready for B2 verification!** ✅

---

**下载链接（如果上面的 AdaCore 下载页面不方便访问）**:

**Direct Link (x86_64 macOS)**:
https://github.com/AdaCore/gnat-ce/releases

**或者告诉我你的 Mac 架构（Intel 还是 Apple Silicon），我给你精确链接！**
