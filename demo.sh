#!/usr/bin/env bash
# Runs the reference 3.5-bit kernel: exhaustive/bit-exact tests, then a CPU micro-benchmark.
# Prints what was measured on this machine; no numbers are hard-coded.
set -euo pipefail
cd "$(dirname "$0")"
command -v gfortran >/dev/null || { echo "gfortran not found (brew install gcc)"; exit 1; }
make -C kernel3p5 test
make -C kernel3p5 bench
