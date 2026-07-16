"""
simulation_01_exact_demod_and_throughput.py - Study 1 (Episode I).

Backs claims C1 (the closed-form hexagonal demodulator returns the exact ML
decision), C2 (its per-symbol cost is in the same O(1) class as a square-QAM
slicer), and C12 (with the exact-predicate path, decode decisions are a provable,
platform-independent function of the received-sample bits).

What it measures
----------------
1. Correctness (C1): over a dense deterministic grid and over noisy streams, the
   closed-form decoder and the exact-predicate referee both agree with exhaustive
   ML on every symbol. Any disagreement is a hard failure.

2. Cost class (C2): median per-symbol decode time for three O(1)/amortized-O(1)
   decoders -- the hexagonal fast-path decoder, the square-QAM independent-axis
   slicer, and (for reference/exactness) exhaustive ML -- across M in {16, 64,
   256, 1024}, using a fixed batching and median-over-repeats protocol. The
   headline is the hex:slicer ratio (parity of cost class), with the exhaustive
   ML column kept as the fallback/exactness referee, not the opponent.

3. Fast-path coverage: the fraction of symbols the hexagonal interior fast path
   resolves without the O(M) exterior fallback, as a function of Eb/N0.

4. Exactness escalation (C12): the fraction of symbols whose float filter cannot
   certify the nearest-point sign and must escalate to the exact Z[sqrt(3)]
   predicate, versus Eb/N0. Small and measured, this is the price of a provably
   exact, bit-reproducible decision rather than a float verdict.

Outputs
-------
  sim01_latency.csv          per-M median ns/symbol for hex / slicer / ML + ratios
  sim01_coverage_vs_snr.csv  fast-path and escalation fractions vs Eb/N0
  sim01_provenance.json      seed, versions, protocol constants

Reproduce: python simulation_01_exact_demod_and_throughput.py

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import json
import platform
import time
from typing import Dict, List

import numpy as np

import tqf_hex_signal as h

SEED = 42
M_VALUES = [16, 64, 256, 1024]
EBN0_TIMING_DB = 15.0          # high SNR so timing reflects the interior fast path
EBN0_COVERAGE_DB = [6.0, 9.0, 12.0, 15.0]
TIMING_SYMBOLS = 200_000
COVERAGE_SYMBOLS = 200_000
ESCALATION_SYMBOLS = 100_000   # exact-predicate path is per-symbol Python; keep modest
TIMING_REPEATS = 7             # median over repeats to suppress scheduler jitter


def _median_ns_per_symbol(fn, n_symbols: int, repeats: int) -> Dict[str, float]:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) / n_symbols * 1e9)
    arr = np.array(times)
    return {"median": float(np.median(arr)),
            "min": float(arr.min()), "max": float(arr.max())}


def run_latency(rng: np.random.Generator) -> List[dict]:
    rows = []
    for M in M_VALUES:
        hexc = h.build_filled_constellation(M)
        hx = h.make_hex_decode_context(hexc)
        tx = rng.integers(0, hexc.size, size=TIMING_SYMBOLS)
        rx_hex = h.awgn(hexc.points_unit[tx], EBN0_TIMING_DB,
                        hexc.bits_per_symbol, rng)

        # warm up JIT-free numpy paths and validate correctness before timing
        idx_fast, fast_frac = h.decode_hex_fast(rx_hex, hx)
        idx_ml = h.decode_ml(rx_hex, hexc.points_unit)
        assert np.array_equal(idx_fast, idx_ml), f"hex fast != ML at M={M}"

        hex_t = _median_ns_per_symbol(lambda: h.decode_hex_fast(rx_hex, hx),
                                      TIMING_SYMBOLS, TIMING_REPEATS)
        ml_t = _median_ns_per_symbol(lambda: h.decode_ml(rx_hex, hexc.points_unit),
                                     TIMING_SYMBOLS, TIMING_REPEATS)

        row = {"M": M,
               "ns_per_symbol_hex": hex_t["median"],
               "hex_ns_min": hex_t["min"], "hex_ns_max": hex_t["max"],
               "ns_per_symbol_ml": ml_t["median"],
               "ml_ns_min": ml_t["min"], "ml_ns_max": ml_t["max"],
               "fast_path_fraction": float(fast_frac.mean()),
               "ebn0_db": EBN0_TIMING_DB}

        # Square-QAM slicer baseline only where M is an even power of two.
        nbits = int(round(np.log2(M)))
        if (1 << nbits) == M and nbits % 2 == 0:
            qam = h.build_square_qam(M)
            qx = h.make_qam_slicer_context(qam)
            txq = rng.integers(0, qam.size, size=TIMING_SYMBOLS)
            rx_q = h.awgn(qam.points_unit[txq], EBN0_TIMING_DB,
                          qam.bits_per_symbol, rng)
            idx_sl = h.decode_qam_slicer(rx_q, qx)
            idx_qml = h.decode_ml(rx_q, qam.points_unit)
            assert np.array_equal(idx_sl, idx_qml), f"QAM slicer != ML at M={M}"
            sl_t = _median_ns_per_symbol(lambda: h.decode_qam_slicer(rx_q, qx),
                                         TIMING_SYMBOLS, TIMING_REPEATS)
            row["ns_per_symbol_qam_slicer"] = sl_t["median"]
            row["qam_ns_min"] = sl_t["min"]
            row["qam_ns_max"] = sl_t["max"]
            row["hex_over_slicer"] = hex_t["median"] / sl_t["median"]
        else:
            row["ns_per_symbol_qam_slicer"] = float("nan")
            row["qam_ns_min"] = float("nan")
            row["qam_ns_max"] = float("nan")
            row["hex_over_slicer"] = float("nan")

        row["ml_over_hex"] = ml_t["median"] / hex_t["median"]
        rows.append(row)
    return rows


def run_coverage(rng: np.random.Generator) -> List[dict]:
    rows = []
    for M in M_VALUES:
        hexc = h.build_filled_constellation(M)
        hx = h.make_hex_decode_context(hexc)
        for ebn0 in EBN0_COVERAGE_DB:
            tx = rng.integers(0, hexc.size, size=COVERAGE_SYMBOLS)
            rx = h.awgn(hexc.points_unit[tx], ebn0, hexc.bits_per_symbol, rng)
            _, fast_frac = h.decode_hex_fast(rx, hx)

            # Exact-predicate escalation rate on a subsample (per-symbol path).
            sub = rx[:ESCALATION_SYMBOLS]
            ref, esc = h.decode_hex_exact_referee(sub, hx, use_filter=True)
            ml_sub = h.decode_ml(sub, hexc.points_unit)
            assert np.array_equal(ref, ml_sub), f"exact referee != ML M={M} {ebn0}dB"

            rows.append({"M": M, "ebn0_db": ebn0,
                         "fast_path_fraction": float(fast_frac.mean()),
                         "fallback_fraction": float(1.0 - fast_frac.mean()),
                         "escalation_fraction": float(esc / sub.shape[0]),
                         "escalation_symbols": int(sub.shape[0])})
    return rows


def _dense_grid_correctness(rng: np.random.Generator) -> int:
    """Decode a dense grid over the M=64 constellation footprint; return mismatches."""
    hexc = h.build_filled_constellation(64)
    hx = h.make_hex_decode_context(hexc)
    lo = hexc.points_unit.real.min() - 0.3, hexc.points_unit.imag.min() - 0.3
    hi = hexc.points_unit.real.max() + 0.3, hexc.points_unit.imag.max() + 0.3
    xs = np.linspace(lo[0], hi[0], 400)
    ys = np.linspace(lo[1], hi[1], 400)
    gx, gy = np.meshgrid(xs, ys)
    grid = (gx + 1j * gy).ravel()
    ml = h.decode_ml(grid, hexc.points_unit)
    fast, _ = h.decode_hex_fast(grid, hx)
    ref, _ = h.decode_hex_exact_referee(grid, hx, use_filter=True)
    return int((fast != ml).sum() + (ref != ml).sum())


def main() -> None:
    rng = np.random.default_rng(SEED)

    mismatches = _dense_grid_correctness(rng)
    assert mismatches == 0, f"grid correctness failed: {mismatches} mismatches"

    latency = run_latency(rng)
    coverage = run_coverage(rng)

    # write CSVs
    import csv
    with open("sim01_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(latency[0].keys()))
        w.writeheader()
        w.writerows(latency)
    with open("sim01_coverage_vs_snr.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage[0].keys()))
        w.writeheader()
        w.writerows(coverage)

    provenance = {
        "study": 1,
        "seed": SEED,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tqf_hex_signal_version": h.__version__,
        "timing_symbols": TIMING_SYMBOLS,
        "timing_repeats": TIMING_REPEATS,
        "coverage_symbols": COVERAGE_SYMBOLS,
        "escalation_symbols": ESCALATION_SYMBOLS,
        "ebn0_timing_db": EBN0_TIMING_DB,
        "ebn0_coverage_db": EBN0_COVERAGE_DB,
        "grid_mismatches": mismatches,
    }
    with open("sim01_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print("Study 1 complete.")
    print(f"  dense-grid mismatches (hex+exact vs ML): {mismatches}")
    for r in latency:
        print(f"  M={r['M']:>4}  hex={r['ns_per_symbol_hex']:.1f} ns  "
              f"slicer={r['ns_per_symbol_qam_slicer']:.1f} ns  "
              f"hex/slicer={r['hex_over_slicer']:.2f}  "
              f"ML/hex={r['ml_over_hex']:.1f}  "
              f"fast={r['fast_path_fraction']*100:.2f}%")


if __name__ == "__main__":
    main()
