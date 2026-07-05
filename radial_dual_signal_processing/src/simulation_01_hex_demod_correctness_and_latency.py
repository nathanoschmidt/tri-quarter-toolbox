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

Timing methodology
------------------
* TQF and ML repeats are INTERLEAVED (A/B, A/B, ...) on the same batch rather
  than timing all-TQF-then-all-ML, so slow environmental drift (thermal/boost
  states, background load) cannot bias one column.
* Every timing is reported as median [min, max] over the repeats; the min/max
  dispersion columns are persisted in the CSV.
* The TQF per-symbol cost is near-flat in M; read its shape against the
  [min, max] dispersion rather than as a trend. The hardware-independent C2
  claim is the growth of the ML/TQF speedup with M, not the fine structure of
  the TQF column.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.2.0
Date: July 4, 2026
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


def _stats_ns(samples: list[float]) -> tuple[float, float, float]:
    """(median, min, max) of a per-symbol nanosecond sample list."""
    arr = np.asarray(samples, dtype=float)
    return float(np.median(arr)), float(np.min(arr)), float(np.max(arr))


def measure_latency(m: int, batch: int, repeats: int, ebn0_db: float,
                    rng: np.random.Generator) -> dict:
    """Interleaved (A/B) timing of the TQF and ML decoders on one shared batch.

    Both decoders are warmed once, then each repeat times TQF immediately
    followed by ML on the SAME received batch, so any slow environmental drift
    hits both columns equally instead of biasing whichever ran second. Returns
    median/min/max ns-per-symbol for each decoder plus the fast-path fraction.
    """
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    tx = rng.integers(0, con.size, batch)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    _idx, fast_mask = t.decode_hex_fast(rx, ctx)
    fast_frac = float(np.mean(fast_mask))

    # warm-up both paths (primes caches/allocations; there is no JIT here)
    t.decode_hex_fast(rx, ctx)
    t.decode_ml(rx, con.points_unit)

    tqf_ns: list[float] = []
    ml_ns: list[float] = []
    for _ in range(repeats):                       # interleaved A/B repeats
        t0 = time.perf_counter()
        t.decode_hex_fast(rx, ctx)
        t1 = time.perf_counter()
        t.decode_ml(rx, con.points_unit)
        t2 = time.perf_counter()
        tqf_ns.append((t1 - t0) / batch * 1e9)
        ml_ns.append((t2 - t1) / batch * 1e9)

    tqf_med, tqf_min, tqf_max = _stats_ns(tqf_ns)
    ml_med, ml_min, ml_max = _stats_ns(ml_ns)
    return {
        "ns_per_symbol_tqf": tqf_med, "tqf_ns_min": tqf_min, "tqf_ns_max": tqf_max,
        "ns_per_symbol_ml": ml_med, "ml_ns_min": ml_min, "ml_ns_max": ml_max,
        "fast_path_fraction": fast_frac,
    }


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
          f"(median [min, max] over {args.timing_repeats} INTERLEAVED A/B repeats, "
          f"batch={args.latency_batch}, Eb/N0={args.ebn0} dB)")
    print("  NOTE: this is batched-NumPy throughput per symbol, not single-symbol "
          "latency.")
    print(f"{'M':>5} {'TQF ns/sym [min, max]':>28} {'ML ns/sym [min, max]':>28} "
          f"{'speedup':>9} {'fast%':>8}")
    rows: List[tuple] = []
    speedups = []
    tqf_meds = []
    for m in args.M:
        r = measure_latency(m, args.latency_batch, args.timing_repeats,
                            args.ebn0, rng)
        speedup = (r["ns_per_symbol_ml"] / r["ns_per_symbol_tqf"]
                   if r["ns_per_symbol_tqf"] > 0 else float("nan"))
        speedups.append(speedup)
        tqf_meds.append(r["ns_per_symbol_tqf"])
        rows.append((m, r["ns_per_symbol_tqf"], r["ns_per_symbol_ml"], speedup,
                     r["fast_path_fraction"], args.ebn0,
                     r["tqf_ns_min"], r["tqf_ns_max"],
                     r["ml_ns_min"], r["ml_ns_max"]))
        print(f"{m:>5} {r['ns_per_symbol_tqf']:>10.2f} "
              f"[{r['tqf_ns_min']:>8.2f}, {r['tqf_ns_max']:>8.2f}] "
              f"{r['ns_per_symbol_ml']:>10.2f} "
              f"[{r['ml_ns_min']:>8.2f}, {r['ml_ns_max']:>8.2f}] "
              f"{speedup:>8.2f}x {r['fast_path_fraction'] * 100:>7.2f}%")
    monotonic = all(speedups[i] <= speedups[i + 1] + 1e-9
                    for i in range(len(speedups) - 1))
    crossover = next((m for m, s in zip(args.M, speedups) if s >= 1.0), None)
    tqf_band = (min(tqf_meds), max(tqf_meds)) if tqf_meds else (0.0, 0.0)
    print("\n  C2 NOTE: ML cost ~ O(M); worst case for TQF is O(M) (the exterior")
    print("  fallback), typical case O(1) when fast-path coverage ~ 1. The measured")
    print(f"  TQF median sits in a {tqf_band[0]:.0f}-{tqf_band[1]:.0f} ns/sym band "
          f"across M -- read its shape")
    print("  against the [min, max] dispersion above rather than as a trend; the")
    print("  hardware-independent C2 claim is the speedup's growth with M, not the")
    print("  fine structure of the near-flat TQF column.")
    print(f"  Speedup monotonic in M: {monotonic}; TQF first wins (speedup>=1) at "
          f"M={crossover} (slower below that).")

    csv_path = os.path.join(args.results_dir, "sim01_latency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ns_per_symbol_tqf", "ns_per_symbol_ml", "speedup",
                    "fast_path_fraction", "ebn0_db",
                    "tqf_ns_min", "tqf_ns_max", "ml_ns_min", "ml_ns_max"])
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
