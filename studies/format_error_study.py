#!/usr/bin/env python3
"""Weight-quantization error of low-bit formats on synthetic weights, at honest bit budgets.

Counts scale overhead in effective bits/weight. Synthetic data only: no model accuracy is
measured, and heavy-tailed weights are a stand-in for real LLM weight statistics.
Reproduce: python3 studies/format_error_study.py   (fixed seed, numpy only)
"""
import json, sys
import numpy as np

rng = np.random.default_rng(20261022)
N = 1 << 20

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
E2M1_FULL = np.concatenate([-E2M1[::-1], E2M1])          # 16 grid points

def nearest(x, grid):
    idx = np.searchsorted(grid, x)
    idx = np.clip(idx, 1, len(grid) - 1)
    lo, hi = grid[idx - 1], grid[idx]
    return np.where(np.abs(x - lo) <= np.abs(x - hi), lo, hi)

def blocks(w, g): return w.reshape(-1, g)

def q_int(w, bits, g, scale_bits=16):
    b = blocks(w, g); qmax = 2 ** (bits - 1) - 1
    s = np.abs(b).max(1, keepdims=True) / qmax; s[s == 0] = 1
    q = np.clip(np.round(b / s), -qmax - 1, qmax) * s
    return q.ravel(), bits + scale_bits / g

def q_mxfp4(w, g=32):                                      # OCP MX: E8M0 scale, block 32
    b = blocks(w, g); amax = np.abs(b).max(1, keepdims=True); amax[amax == 0] = 1
    s = 2.0 ** (np.floor(np.log2(amax)) - 2)
    q = nearest(np.clip(b / s, -6, 6), E2M1_FULL) * s
    return q.ravel(), 4 + 8 / g

def q_nvfp4(w, g=16):                                      # E4M3 block scale (3-bit mantissa approx)
    b = blocks(w, g); amax = np.abs(b).max(1, keepdims=True); amax[amax == 0] = 1
    s = amax / 6
    e = np.floor(np.log2(s)); m = np.round(s / 2.0 ** e * 8) / 8
    s = m * 2.0 ** e
    q = nearest(np.clip(b / s, -6, 6), E2M1_FULL) * s
    return q.ravel(), 4 + 8 / g

def q_pair35(w, g=32, scale_bits=16):                      # repo format: even 4-bit, odd 3-bit, one scale/block
    b = blocks(w, g); out = np.empty_like(b)
    lev_e = np.arange(-8, 8); lev_o = np.arange(-4, 4)
    amax = np.abs(b).max(1, keepdims=True); amax[amax == 0] = 1
    best_err = np.full(len(b), np.inf)
    for f in np.linspace(0.15, 1.0, 60):                   # per-block scale search
        s = amax * f / 7
        q = np.empty_like(b)
        q[:, 0::2] = np.clip(np.round(b[:, 0::2] / s), -8, 7) * s
        q[:, 1::2] = np.clip(np.round(b[:, 1::2] / s), -4, 3) * s
        err = ((q - b) ** 2).sum(1)
        upd = err < best_err
        out[upd] = q[upd]; best_err[upd] = err[upd]
    return out.ravel(), 3.5 + scale_bits / g

def sqnr_db(w, q): return 10 * np.log10((w ** 2).sum() / ((w - q) ** 2).sum())

dists = {
    "gaussian": rng.standard_normal(N),
    "student_t3 (heavy tail)": rng.standard_t(3, N) / np.sqrt(3),
    "gaussian + 0.1% 10x outliers": np.where(rng.random(N) < 0.001, 10, 1) * rng.standard_normal(N),
}
formats = {
    "INT4 g128 (fp16 scale)": lambda w: q_int(w, 4, 128),
    "INT4 g32  (fp16 scale)": lambda w: q_int(w, 4, 32),
    "MXFP4 g32 (E8M0)":       q_mxfp4,
    "NVFP4 g16 (E4M3)":       q_nvfp4,
    "3.5-bit pair g32":       lambda w: q_pair35(w, 32),
    "3.5-bit pair g128":      lambda w: q_pair35(w, 128),
    "INT3 g128 (fp16 scale)": lambda w: q_int(w, 3, 128),
}
res = {}
for dn, w in dists.items():
    res[dn] = {}
    for fn, f in formats.items():
        q, bits = f(w)
        res[dn][fn] = {"bits_per_weight": round(bits, 3), "sqnr_db": round(float(sqnr_db(w, q)), 2)}

if "--json" in sys.argv: print(json.dumps(res, indent=1)); sys.exit()
for dn, rows in res.items():
    print(f"\n{dn}  (N={N})")
    print(f"  {'format':26s} {'bits/w':>7s} {'SQNR dB':>8s}")
    for fn, r in sorted(rows.items(), key=lambda kv: kv[1]['bits_per_weight']):
        print(f"  {fn:26s} {r['bits_per_weight']:7.3f} {r['sqnr_db']:8.2f}")
