# Contributing to 3.5-bit Quantized LLM Inference

Thank you for your interest in contributing! This project aims to push the boundaries of LLM inference efficiency through novel quantization schemes and hardware optimization.

## 🎯 Areas of Contribution

### 1. Hardware Testing
**Priority: HIGH**

We need help testing on real Groq LPU hardware:
- Run the compiled binaries on actual Groq LPUs
- Benchmark throughput and validate 10,000+ tok/s target
- Profile memory bandwidth utilization
- Report any performance issues or optimization opportunities

**Requirements**:
- Access to Groq hardware or cloud instances
- Familiarity with MLIR and ASIC deployment
- Experience with performance profiling

### 2. Formal Verification
**Priority: MEDIUM**

Help complete the Lean 4 proofs:
- Fill in `sorry` placeholders in `ErrorBounds.lean`
- Prove `quantization_error_bound` using Mathlib floor/ceil lemmas
- Prove `no_int32_overflow` using summation bounds
- Add new theorems for numerical stability

**Requirements**:
- Experience with Lean 4 and Mathlib4
- Understanding of quantization error analysis
- Familiarity with formal verification

**Getting started**:
```bash
cd lean-verification
lake build
code Quantization3p5bit/ErrorBounds.lean
```

### 3. Performance Optimization
**Priority: MEDIUM**

Further optimize the Fortran implementation:
- GPU kernels (CUDA/ROCm) for comparison
- Additional SIMD optimizations (AVX-512, SVE)
- Memory layout improvements
- Alternative quantization schemes (2-bit, 5-bit, etc.)

**Requirements**:
- Strong Fortran or C/C++ skills
- SIMD programming experience
- Profiling and optimization expertise

### 4. Model Support
**Priority: LOW**

Extend to additional model architectures:
- Mistral 7B/8x7B
- Gemma 2B/7B
- Phi-3
- Qwen models
- Custom architectures

**Requirements**:
- Understanding of transformer architectures
- Experience with model weights and quantization
- Python/Fortran interoperability

### 5. Documentation
**Priority: ONGOING**

Improve documentation and examples:
- Add more code comments
- Create tutorial notebooks
- Write blog posts explaining the approach
- Improve README clarity
- Add diagrams and visualizations

**Requirements**:
- Good technical writing skills
- Understanding of quantization concepts
- Markdown/LaTeX proficiency

## 🔧 Development Setup

### Prerequisites

```bash
# macOS
brew install gcc  # gfortran 13.2+
brew install llvm  # For MLIR tools (optional)

# Linux
sudo apt install gfortran build-essential
```

### Clone and Build

```bash
# Clone repository
git clone https://github.com/yourusername/3.5bit-groq-mvp.git
cd 3.5bit-groq-mvp/2025-3.5bit-groq-mvp

# Build all targets
make clean
make all

# Run tests
make test

# Run benchmarks
make benchmark-simd
```

### Lean 4 Setup (for verification work)

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y

# Build Lean project
cd ../lean-verification
lake update
lake build
```

## 📝 Contribution Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/3.5bit-groq-mvp.git
cd 3.5bit-groq-mvp
git remote add upstream https://github.com/ORIGINAL_OWNER/3.5bit-groq-mvp.git
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions**:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation improvements
- `perf/` - Performance optimizations
- `proof/` - Lean verification work

### 3. Make Changes

Follow our coding standards:

**Fortran Code**:
- Use Fortran 2023 standard features
- Follow `do concurrent` for parallelism where possible
- Include comments explaining algorithm choices
- Add unit tests for new functionality
- Run `make test` before committing

**Lean Code**:
- Follow Mathlib4 style guide
- Document theorems with `/-! ... -/` comments
- Avoid `sorry` in final submissions (unless documented as TODO)
- Run `lake build` to ensure no errors

**Documentation**:
- Use clear, concise language
- Include code examples where applicable
- Update README.md if adding major features
- Check for typos and formatting

### 4. Test Your Changes

```bash
# Run all tests
make test

# Run benchmarks
make benchmark-simd

# Build Lean proofs
cd ../lean-verification && lake build

# Check for regressions
./scripts/run_all_tests.sh
```

### 5. Commit and Push

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: Add GPU kernel for quantized matmul

- Implement CUDA kernel for INT4 quantization
- Achieve 2.3× speedup over CPU SIMD version
- Add comprehensive benchmarks and unit tests"

# Push to your fork
git push origin feature/your-feature-name
```

**Commit message format**:
```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

**Types**: `feat`, `fix`, `docs`, `perf`, `test`, `refactor`, `chore`

### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select `main` as the base branch
4. Fill in the PR template with:
   - Clear description of changes
   - Related issue numbers (if applicable)
   - Testing performed
   - Benchmarks (if performance-related)
   - Screenshots/outputs (if applicable)

### 7. Code Review

- Respond to reviewer comments
- Make requested changes
- Push updates to the same branch
- Be open to feedback and suggestions

## 🧪 Testing Guidelines

### Unit Tests

Add tests for new functionality:

```fortran
program test_new_feature
  use matmul_int4_groq, only: new_function
  implicit none
  
  real(sp) :: result, expected
  
  ! Test case 1
  result = new_function(input1)
  expected = 42.0
  if (abs(result - expected) > 1.0e-6) then
    print *, "FAIL: Test case 1"
    stop 1
  end if
  
  print *, "✓ All tests passed"
end program
```

### Benchmark Tests

Include performance measurements:

```fortran
! Before optimization: 100 ms
! After optimization: 43 ms (2.3× speedup)
! Hardware: M1 Max, 4 threads
```

### Lean Proofs

Ensure all theorems build:

```bash
cd lean-verification
lake build --verbose
# Should complete without errors
```

## 📊 Performance Requirements

Pull requests introducing performance changes should include:

1. **Benchmark results** comparing before/after
2. **Hardware specifications** used for testing
3. **Compiler flags** and environment details
4. **Correctness verification** (bit-exact or RMSE)

**Minimum performance criteria**:
- Must not regress existing benchmarks by >5%
- Must maintain bit-exact correctness (or document RMSE)
- Must work on both macOS and Linux

## 🐛 Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Reproduction**: Steps to reproduce the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - OS and version
   - Compiler and version
   - Hardware (CPU, RAM)
6. **Logs**: Relevant error messages or logs

**Example**:

```markdown
### Bug: SIMD optimization crashes on AVX-512

**Description**: The SIMD-optimized matmul crashes when compiled with AVX-512 flags.

**Reproduction**:
1. `make clean`
2. `make FFLAGS="-O3 -march=native" benchmark-simd`
3. Run `./bench_simd`

**Expected**: Benchmark completes successfully
**Actual**: Segmentation fault at line 127

**Environment**:
- OS: Ubuntu 22.04
- Compiler: gfortran 13.2
- CPU: Intel Xeon with AVX-512

**Logs**:
```
Program received signal SIGSEGV: Segmentation fault
Backtrace for this error:
...
```
```

## 💡 Feature Requests

Propose new features via GitHub Issues:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How would it work?
3. **Alternatives**: What other approaches were considered?
4. **Impact**: What would this enable?

## 📜 Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Assume good intent
- Prioritize project goals over personal preferences

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 📞 Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@example.com (private disclosure)
- **Chat**: Join our Discord/Slack (link TBD)

## 🎓 Resources

### Learn About Quantization
- [AWQ Paper](https://arxiv.org/abs/2306.00978) - Activation-aware quantization
- [GPTQ Paper](https://arxiv.org/abs/2210.17323) - Post-training quantization

### Learn Fortran
- [Modern Fortran](https://fortran-lang.org/learn/) - Official tutorials
- [Fortran 2023 Standard](https://j3-fortran.org/) - Language specification

### Learn Lean 4
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)

### Learn MLIR
- [MLIR Documentation](https://mlir.llvm.org/)
- [Groq LPU Guide](https://groq.com/developers)

## 🙏 Thank You!

Your contributions help advance the state of efficient LLM inference. Every contribution, no matter how small, is valuable and appreciated!

---

**Questions?** Open a GitHub Discussion or reach out to the maintainers.

Last updated: 2025-11-28
