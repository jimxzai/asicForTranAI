#!/usr/bin/env bash
# Claims audit: checks the project's headline claims against what the repo can actually show.
# Read-only. Run from anywhere: ./scripts/claims_audit.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1
fail=0
flag() { echo "  FAIL: $1"; fail=$((fail + 1)); }
ok()   { echo "  ok:   $1"; }

echo "1-2. Claims (19 GB, 4,188 tok/s, 247 proofs, 46% smaller) repeated without a warning"
python3 - <<'PYEOF'
import re, subprocess, sys
pat = re.compile(r'19 ?GB|4,?188|\b247\b|46 ?%')
marker = re.compile(r'Archived / unaudited|Superseded 2026|NOTE \(2026-10-22\)|DRAFT, NOT FOR SUBMISSION|CLAIMS_AUDIT')
discuss = {'scripts/claims_audit.sh', 'docs/CLAIMS_AUDIT_2026-10.md', 'docs/EVALUATION_2026-10.md', 'docs/outreach/DRAFTS_2026-10.md', 'spark-llama-safety/README.md'}  # quote the claims in order to refute them
bad = []
for f in subprocess.run(['git', 'ls-files'], capture_output=True, text=True).stdout.splitlines():
    if f in discuss or f.startswith('docs/archive/') or f.endswith(('.pdf', '.png', '.json', '.lock', '.safetensors', '.model')):
        continue
    try: s = open(f, errors='ignore').read()
    except OSError: continue
    if pat.search(s) and not marker.search(s): bad.append(f)
print(f"  files repeating a claim with no warning marker: {len(bad)}")
for f in bad[:15]: print("    ", f)
sys.exit(1 if bad else 0)
PYEOF
[ $? -ne 0 ] && flag "claims repeated without a warning marker (list above)"
python3 -c "
params = 70.6e9
print('  3.5-bit weights alone: %.1f GB; 4-bit: %.1f GB; 19 GB would be %.2f bits/param' % (params*3.5/8/1e9, params*4/8/1e9, 19e9*8/params))"

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
for d in lean-verification lean-alphaproof-mcts; do
  grep -q 'contain unproven goals' "$d/README.md" 2>/dev/null || flag "$d has unproven goals but no README warning"
done
[ "$kern" -gt 0 ] && flag "$kern 'sorry' in lean-kernel"
if command -v lake >/dev/null 2>&1; then
  if lb=$(cd lean-kernel && lake build 2>&1); then ok "lean-kernel builds: $(echo "$lb" | tail -1)"; else flag "lean-kernel build fails"; fi
else
  flag "lake not installed here (see docs/INSTALL_GNAT.md), so Lean proofs cannot be reproduced"
fi

echo "5. Legacy kernel storage: '3.5-bit' format in matmul_3p5bit_dynamic.f90"
if grep -qE 'W_Q\(K/2, N\)' 2025-3.5bit-groq-mvp/matmul_3p5bit_dynamic.f90 2>/dev/null; then
  if grep -q 'NOTE (2026-10-22)' 2025-3.5bit-groq-mvp/matmul_3p5bit_dynamic.f90; then
    ok "legacy kernel stores 4 bits/weight (one byte per pair) and says so in its header; use kernel3p5/"
  else
    flag "legacy kernel stores W_Q(K/2,N) as int8 = 4 bits/weight and is not labelled"
  fi
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
