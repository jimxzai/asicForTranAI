#!/usr/bin/env bash
# Wraps the benchmark with the context needed to reproduce or discount a result.
set -eu
cpu=$( (sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | sed 's/^ *//')
echo "{ \"date\": \"$(date -u +%FT%TZ)\", \"cpu\": \"${cpu}\", \"compiler\": \"$(gfortran --version | head -1)\","
echo "  \"flags\": \"-O3 -march=native\", \"threads\": 1, \"result\":"
"$@"
echo "}"
