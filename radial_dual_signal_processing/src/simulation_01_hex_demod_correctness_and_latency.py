#!/usr/bin/env python3
"""
simulation_01_hex_demod_correctness_and_latency.py - Exact Hexagonal Demodulation Correctness and Latency (C1, C2)

Validates the exactness and constant-time-decode claims of the Tri-Quarter
Framework (TQF) radial_dual_signal_processing subproject.

Claims validated
----------------
C1 (Exactness): the closed-form constant-time hexagonal demodulator produces
    decisions that are bitwise-identical to exhaustive maximum-likelihood (ML)
    nearest-point decoding -- verified on a dense deterministic grid that tiles
    the constellation region and on a Monte-Carlo channel stream (zero
    mismatches expected).
C2 (Constant-time decode): the closed-form decoder is O(1) per symbol on its
    fast (interior) path -- a fixed 3x3 candidate window -- versus O(M) for
    exhaustive ML, so the measured per-symbol throughput speedup grows
    monotonically with M above a small crossover. The O(1) path is NOT
    universal: exterior symbols fall back to O(M) ML, so the worst case is O(M)
    and the amortized cost is contingent on the fast-path coverage, which this
    script reports as a function of SNR.

Output: the two PASS/FAIL banners and the latency table.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.1.0
Date: June 27, 2026
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import List

import numpy as np

import tqf_hex_signal as t


def verify_dense_grid(m: int, step: float, margin: float) -> tuple[int, int, float]:
    """Decode every point of a fine grid covering the constellation region with
    both decoders and count mismatches.

    Returns (num_points, num_mismatches, fast_path_fraction).
    """
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    re = con.points_unit.real
    im = con.points_unit.imag
    xs = np.arange(re.min() - margin, re.max() + margin, step)
    ys = np.arange(im.min() - margin, im.max() + margin, step)
    gx, gy = np.meshgrid(xs, ys)
    grid = (gx + 1j * gy).ravel()

    dec_fast, fast_mask = t.decode_hex_fast(grid, ctx)
    dec_ml = t.decode_ml(grid, con.points_unit)
    mismatches = int(np.sum(dec_fast != dec_ml))
    return grid.size, mismatches, float(np.mean(fast_mask))


def verify_monte_carlo(m: int, trials: int, ebn0_db: float,
                       rng: np.random.Generator) -> tuple[int, float]:
    """Decode a noisy channel stream with both decoders; return (mismatches,
    fast_path_fraction)."""
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    tx = rng.integers(0, con.size, trials)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    dec_fast, fast_mask = t.decode_hex_fast(rx, ctx)
    dec_ml = t.decode_ml(rx, con.points_unit)
    return int(np.sum(dec_fast != dec_ml)), float(np.mean(fast_mask))


def _median_ns_per_symbol(decode_call, n_symbols: int, repeats: int) -> float:
    """Time a decode call over a fixed batch and return median ns per symbol."""
    decode_call()  # warm-up (JIT-free here, but primes caches/allocations)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decode_call()
        t1 = time.perf_counter()
        samples.append((t1 - t0) / n_symbols * 1e9)
    return float(np.median(samples))


def measure_latency(m: int, batch: int, repeats: int, ebn0_db: float,
                    rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (ns/symbol TQF, ns/symbol ML, fast-path fraction) for one batch."""
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    tx = rng.integers(0, con.size, batch)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    _idx, fast_mask = t.decode_hex_fast(rx, ctx)
    fast_frac = float(np.mean(fast_mask))
    ns_tqf = _median_ns_per_symbol(lambda: t.decode_hex_fast(rx, ctx), batch, repeats)
    ns_ml = _median_ns_per_symbol(lambda: t.decode_ml(rx, con.points_unit), batch, repeats)
    return ns_tqf, ns_ml, fast_frac


def fast_path_fraction(m: int, trials: int, ebn0_db: float,
                       rng: np.random.Generator) -> float:
    """Fraction of symbols decoded by the O(1) window (no O(M) ML fallback)."""
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    tx = rng.integers(0, con.size, trials)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    _idx, fast_mask = t.decode_hex_fast(rx, ctx)
    return float(np.mean(fast_mask))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, nargs="+", default=[16, 64, 256],
                    help="constellation orders (powers of two)")
    ap.add_argument("--trials", type=int, default=100_000,
                    help="Monte-Carlo symbols for the correctness stream")
    ap.add_argument("--ebn0", type=float, default=12.0,
                    help="Eb/N0 (dB) for the correctness stream and latency batch")
    ap.add_argument("--grid_step", type=float, default=0.01,
                    help="dense-grid spacing in signal-space units")
    ap.add_argument("--grid_margin", type=float, default=0.3,
                    help="margin added around the constellation for the grid")
    ap.add_argument("--latency_batch", type=int, default=200_000,
                    help="symbols per timed decode batch")
    ap.add_argument("--fastpath_ebn0", type=float, nargs="+", default=[0, 4, 8, 12],
                    help="Eb/N0 grid (dB) for the fast-path coverage sweep")
    ap.add_argument("--timing_repeats", type=int, default=15,
                    help="timed repeats per measurement (median reported)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 72)
    print("SIMULATION 01 -- exact-decode correctness (C1) and latency (C2)")
    print(f"seed={args.seed}  M={args.M}  trials={args.trials}  Eb/N0={args.ebn0} dB")
    print("=" * 72)
    t.emit_provenance(args.results_dir, "sim01", args=args)

    # ---- C1: correctness ---------------------------------------------------
    total_mismatch = 0
    print("\n[C1] Closed-form decoder vs exhaustive ML")
    print(f"{'M':>5} {'grid pts':>10} {'grid mis':>9} {'MC mis':>8} "
          f"{'fast% grid':>11} {'fast% MC':>9}")
    for m in args.M:
        gpts, gmis, gfast = verify_dense_grid(m, args.grid_step, args.grid_margin)
        mcmis, mcfast = verify_monte_carlo(m, args.trials, args.ebn0, rng)
        total_mismatch += gmis + mcmis
        print(f"{m:>5} {gpts:>10} {gmis:>9} {mcmis:>8} "
              f"{gfast * 100:>10.2f}% {mcfast * 100:>8.2f}%")
    c1_pass = (total_mismatch == 0)
    print(f"\n  C1 RESULT: {'PASS' if c1_pass else 'FAIL'} "
          f"(total mismatches across all M = {total_mismatch})")

    # ---- C2 (coverage): O(1) fast-path fraction vs SNR --------------------
    # The O(1) claim is fast-path-contingent: exterior symbols fall back to
    # O(M) ML, and that exterior fraction grows as SNR falls (and as M grows).
    fp_M = max(args.M)  # largest constellation -> worst-case coverage
    print(f"\n[C2 coverage] O(1) fast-path fraction vs Eb/N0 (M={fp_M}); "
          f"exterior symbols use the O(M) ML fallback")
    print(f"{'Eb/N0(dB)':>10} {'fast-path %':>12} {'fallback %':>12}")
    coverage_rows: List[tuple] = []
    for ebn0 in args.fastpath_ebn0:
        frac = fast_path_fraction(fp_M, args.trials, ebn0, rng)
        coverage_rows.append((fp_M, ebn0, frac, 1 - frac))
        print(f"{ebn0:>10.1f} {frac * 100:>11.3f}% {(1 - frac) * 100:>11.3f}%")
    print("  -> report fast-path coverage at YOUR operating SNR; at low SNR the "
          "O(M) fallback dominates.")
    cov_path = os.path.join(args.results_dir, "sim01_coverage_vs_snr.csv")
    with open(cov_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ebn0_db", "fast_path_fraction", "fallback_fraction"])
        w.writerows(coverage_rows)

    # ---- C2 (throughput): amortized per-symbol decode time vs M -----------
    print(f"\n[C2 throughput] Amortized per-symbol decode time "
          f"(median over {args.timing_repeats} repeats, batch={args.latency_batch}, "
          f"Eb/N0={args.ebn0} dB)")
    print("  NOTE: this is batched-NumPy throughput per symbol, not single-symbol "
          "latency.")
    print(f"{'M':>5} {'TQF ns/sym':>12} {'ML ns/sym':>12} {'speedup':>9} {'fast%':>8}")
    rows: List[tuple] = []
    speedups = []
    for m in args.M:
        ns_tqf, ns_ml, fast_frac = measure_latency(m, args.latency_batch,
                                                    args.timing_repeats, args.ebn0, rng)
        speedup = ns_ml / ns_tqf if ns_tqf > 0 else float("nan")
        speedups.append(speedup)
        rows.append((m, ns_tqf, ns_ml, speedup, fast_frac, args.ebn0))
        print(f"{m:>5} {ns_tqf:>12.2f} {ns_ml:>12.2f} {speedup:>8.2f}x "
              f"{fast_frac * 100:>7.2f}%")
    monotonic = all(speedups[i] <= speedups[i + 1] + 1e-9
                    for i in range(len(speedups) - 1))
    crossover = next((m for m, s in zip(args.M, speedups) if s >= 1.0), None)
    print("\n  C2 NOTE: ML cost ~ O(M). TQF is O(1) on the fast path, but its")
    print("  measured cost is NOT strictly flat in M: the O(M) exterior fallback")
    print("  and a larger occupancy grid (cache pressure) make it drift upward.")
    print("  Worst case is O(M); typical case O(1) when fast-path coverage ~ 1.")
    print(f"  Speedup monotonic in M: {monotonic}; TQF first wins (speedup>=1) at "
          f"M={crossover} (slower below that).")

    csv_path = os.path.join(args.results_dir, "sim01_latency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ns_per_symbol_tqf", "ns_per_symbol_ml", "speedup",
                    "fast_path_fraction", "ebn0_db"])
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
