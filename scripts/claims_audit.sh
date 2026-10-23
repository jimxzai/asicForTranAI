#!/usr/bin/env bash
# Claims audit: checks the project's headline claims against what the repo can actually show.
# Read-only. Run from anywhere: ./scripts/claims_audit.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
fail=0
flag() { echo "  FAIL: $1"; fail=$((fail + 1)); }
ok()   { echo "  ok:   $1"; }

echo "1. Model size: '70B in 19 GB at 3.5 bits'"
python3 - <<'EOF'
params = 70.6e9  # Llama-70B parameter count
for bits in (4.0, 3.5):
    print(f"  {bits}-bit weights alone: {params * bits / 8 / 1e9:.1f} GB (decimal)")
print(f"  bits/param implied by 19 GB: {19e9 * 8 / params:.2f}")
EOF
if git grep -qE '19 ?GB' -- '2025-3.5bit-groq-mvp/*.md' 2>/dev/null; then
  flag "19 GB appears in docs; 3.5 bits/param on 70B is ~30.9 GB (its own benchmark report says 32.6 GB)"
fi

echo "2. Throughput: '4188 tok/s on Groq, +28.9% vs INT4'"
python3 -c "print(f'  4188/3124 - 1 = {4188/3124 - 1:.1%} (docs say 28.9%)')"
grep -n 'Estimated on Groq' 2025-3.5bit-groq-mvp/benchmark_report_3p5bit.md >/dev/null 2>&1 \
  && flag "benchmark report labels the Groq numbers 'Estimated'; no measured Groq run is in the repo"
git grep -ln '4,188 tok/s measured' -- 2025-3.5bit-groq-mvp 2>/dev/null \
  | sed 's/^/  doc says "measured": /'

export PATH="$HOME/.alire/bin:$HOME/.elan/bin:$(ls -d "$HOME"/.local/share/alire/toolchains/gnat_native_* 2>/dev/null | head -1)/bin:$(ls -d "$HOME"/.local/share/alire/toolchains/gprbuild_* 2>/dev/null | head -1)/bin:$PATH"

echo "3. SPARK: proofs of the reference code in spark-llama-safety/src"
echo "  Ada/SPARK source in src/: $(find spark-llama-safety/src -name '*.ad[sb]' | wc -l | tr -d ' ') files, $(find spark-llama-safety/src -name '*.ad[sb]' -exec cat {} + | wc -l | tr -d ' ') lines; unproved templates kept in legacy/"
if grep -rq 'placeholder' spark-llama-safety/Makefile 2>/dev/null; then flag "SPARK Makefile contains placeholder output"; fi
if grep -q 'TODO' .github/workflows/verify.yml 2>/dev/null; then flag "CI contains TODO stubs"; fi
if command -v gnatprove >/dev/null 2>&1; then
  if gp=$(cd spark-llama-safety && gnatprove -P transformer.gpr --warnings=continue 2>&1); then
    ok "$(echo "$gp" | grep -E 'Success: all checks proved')"
  else
    flag "gnatprove reports unproved checks or errors: $(echo "$gp" | tail -3 | tr '\n' ' ')"
  fi
else
  flag "gnatprove not installed here (see docs/INSTALL_GNAT.md), so proofs cannot be reproduced"
fi

echo "4. Lean 4: lean-kernel (core Lean) and legacy directories"
legacy=$(git grep -c 'sorry' -- 'lean-verification/*.lean' 'lean-alphaproof-mcts/*.lean' '2025-3.5bit-groq-mvp/lean-proofs/*.lean' 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
kern=$(git grep -c 'sorry' -- 'lean-kernel/*.lean' 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
echo "  'sorry' in lean-kernel: $kern; in legacy Lean directories: $legacy"
[ "$legacy" -gt 0 ] && flag "$legacy 'sorry' in legacy Lean sources (lean-verification, lean-alphaproof-mcts, lean-proofs): not proofs"
[ "$kern" -gt 0 ] && flag "$kern 'sorry' in lean-kernel"
if command -v lake >/dev/null 2>&1; then
  if lb=$(cd lean-kernel && lake build 2>&1); then ok "lean-kernel builds: $(echo "$lb" | tail -1)"; else flag "lean-kernel build fails"; fi
else
  flag "lake not installed here (see docs/INSTALL_GNAT.md), so Lean proofs cannot be reproduced"
fi

echo "5. Legacy kernel storage: '3.5-bit' format in matmul_3p5bit_dynamic.f90"
if grep -qE 'W_Q\(K/2, N\)' 2025-3.5bit-groq-mvp/matmul_3p5bit_dynamic.f90 2>/dev/null; then
  flag "legacy kernel stores W_Q(K/2,N) as int8: one byte per pair of weights = 4 bits/weight, so no saving vs INT4 (use kernel3p5/ for true 3.5-bit packing)"
fi

echo "6. Reference kernel (kernel3p5): tests and measured storage"
if command -v gfortran >/dev/null 2>&1; then
  if make -s -C kernel3p5 test >/tmp/kernel3p5_test.log 2>&1; then
    ok "kernel3p5 tests pass: $(grep -E 'measured storage|ALL TESTS' /tmp/kernel3p5_test.log | tr -s ' ' | paste -sd ';' -)"
  else
    flag "kernel3p5 tests fail; see /tmp/kernel3p5_test.log"
  fi
else
  flag "gfortran not installed; reference kernel tests not run"
fi

echo
echo "Failures: $fail"
exit $((fail > 0))
