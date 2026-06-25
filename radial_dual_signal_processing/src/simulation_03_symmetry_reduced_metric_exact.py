#!/usr/bin/env python3
"""
simulation_03_symmetry_reduced_metric_exact.py - Symmetry-Reduced Exact Constellation Metric (C4)

Computes an exact constellation metric from one representative per order-6 orbit
for the Tri-Quarter Framework (TQF) radial_dual_signal_processing subproject.

Claim validated
---------------
C4 (Symmetry-reduced exact evaluation): an exact, performance-relevant
    constellation metric -- the pairwise squared-distance enumerator, from which
    the minimum distance, its multiplicity (kissing number), and the mean
    squared distance follow -- can be computed from one representative per
    order-6 (Z6) rotation orbit and replicated, reproducing the
    full-constellation value as the *identical* exact integer/rational quantity
    (verified with ``==``, not a tolerance). The distance-evaluation count drops
    by exactly 6x (the exact, hardware-independent headline); the wall-clock time
    is reported only as memory-bound corroboration and may fall below OR above 6x
    (the full M x M tally is more cache-hostile than the reduced (M/6) x M one).
    This is an OFFLINE constellation-design computation, not a receive-path cost.

Why this is exact: on the lattice the squared distance between two points is the
integer Eisenstein norm of their difference, da^2 + da*db + db^2. The metric is
therefore computed in pure integer/rational arithmetic with no floating point.
The 6-fold-symmetric disk constellation is closed under the rotation R, so the
six members of each orbit contribute identical distance multisets to the whole
constellation; summing over one representative per orbit and multiplying by six
reproduces the full enumerator exactly.

What to paste back: the results table (sizes, speedup, exact-match flags).

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.0.0
Date: June 24, 2026
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

import tqf_hex_signal as t


def _pair_enumerator_full(ab: np.ndarray, size: int) -> np.ndarray:
    """Full O(M^2) enumerator of squared distances over all ordered pairs i != j.

    Fully vectorized with no boolean masking: tally the full (M, M) squared-
    distance matrix (including the M diagonal self-distances) with one bincount,
    then remove the diagonal's contribution from the zero bin. Returns a count
    vector indexed by squared distance.
    """
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    m = ab.shape[0]
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    sq = da * da + da * db + db * db          # (M, M) integer squared distances
    counts = np.bincount(sq.ravel(), minlength=size).astype(np.int64)
    counts[0] -= m                            # drop the m diagonal zeros
    return counts


def _orbit_partition(ab: np.ndarray) -> List[int]:
    """Return one representative index per order-6 rotation orbit.

    Each non-origin disk point has a full 6-element orbit; we pick the first-seen
    member as the representative. This is O(M) preprocessing (the symmetry-
    reduction setup), excluded from the timed metric computation, exactly as the
    lattice paper treats its clustering symmetry reduction.
    """
    seen = set()
    reps: List[int] = []
    for i in range(ab.shape[0]):
        key = (int(ab[i, 0]), int(ab[i, 1]))
        if key in seen:
            continue
        ca, cb = key
        orbit = []
        for _ in range(6):
            orbit.append((ca, cb))
            ca, cb = t.rotate60(ca, cb)
        if len(set(orbit)) != 6:
            raise ValueError("non-origin disk point with a short orbit (unexpected)")
        for o in orbit:
            seen.add(o)
        reps.append(i)
    return reps


def _pair_enumerator_orbit(ab: np.ndarray, reps: np.ndarray,
                           size: int) -> np.ndarray:
    """Orbit-reduced enumerator: squared distances from each representative to all
    points, tallied once and multiplied by the orbit size 6, with the
    representatives' self-distances removed from the zero bin.

    Fully vectorized over (num_orbits, M) with no boolean masking. Performs
    exactly M^2 / 6 of the full method's distance computations.
    """
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    r = reps.shape[0]
    da = a[reps][:, None] - a[None, :]
    db = b[reps][:, None] - b[None, :]
    sq = da * da + da * db + db * db          # (num_orbits, M)
    counts = np.bincount(sq.ravel(), minlength=size).astype(np.int64) * 6
    counts[0] -= 6 * r                        # drop the r self-distances (x6)
    return counts


def _metric_summary(enum: np.ndarray, m: int) -> Dict[str, object]:
    """Derive the union-bound-relevant quantities from a count vector enumerator."""
    nz = np.nonzero(enum)[0]
    d_min_sq = int(nz[nz >= 1][0])
    multiplicity = int(enum[d_min_sq])                  # ordered nearest pairs
    total = int(np.dot(np.arange(enum.shape[0]), enum))
    mean_sq = Fraction(total, m * (m - 1))              # exact rational
    return {
        "d_min_sq": d_min_sq,
        "multiplicity": multiplicity,
        "avg_kissing": Fraction(multiplicity, m),
        "mean_sq_dist": mean_sq,
    }


def _time_call(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_norm_sq", type=int, nargs="+",
                    default=[7, 19, 37, 61, 91, 127],
                    help="squared-norm thresholds defining 6-fold-symmetric "
                         "disk constellations of increasing size")
    ap.add_argument("--timing_repeats", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)  # determinism only; unused
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print("=" * 84)
    print("SIMULATION 03 -- symmetry-reduced EXACT constellation metric (C4)")
    print("metric: pairwise squared-distance enumerator (integer/rational, exact)")
    print("=" * 84)
    t.emit_provenance(args.results_dir, "sim03", args=args)
    print(f"{'M':>5} {'orbits':>7} {'dmin^2':>7} {'mult':>6} "
          f"{'evalratio':>9} {'wallspeed':>10} {'exact==':>8}")

    rows: List[tuple] = []
    all_exact = True
    for T in args.max_norm_sq:
        con = t.build_disk_constellation(T)
        ab = con.ab
        m = ab.shape[0]
        reps = np.array(_orbit_partition(ab), dtype=np.int64)
        size = int(4 * T + 1)  # squared distances are bounded by ~4*max_norm_sq

        enum_full = _pair_enumerator_full(ab, size)
        enum_orbit = _pair_enumerator_orbit(ab, reps, size)

        # Exact verification: identical enumerators and identical derived metrics.
        sfull = _metric_summary(enum_full, m)
        sorbit = _metric_summary(enum_orbit, m)
        exact = bool(np.array_equal(enum_full, enum_orbit)) and (sfull == sorbit)
        all_exact = all_exact and exact

        # Exact operation-count reduction (the 6x ceiling, met by construction):
        full_evals = m * (m - 1)
        orbit_evals = reps.shape[0] * (m - 1)
        eval_ratio = full_evals / orbit_evals          # == 6.0 exactly

        full_ms = _time_call(lambda: _pair_enumerator_full(ab, size),
                             args.timing_repeats) * 1e3
        orbit_ms = _time_call(lambda: _pair_enumerator_orbit(ab, reps, size),
                              args.timing_repeats) * 1e3
        wall_speed = full_ms / orbit_ms if orbit_ms > 0 else float("nan")

        print(f"{m:>5} {len(reps):>7} {sfull['d_min_sq']:>7} "
              f"{sfull['multiplicity']:>6} {eval_ratio:>8.2f}x {wall_speed:>9.2f}x "
              f"{str(exact):>8}")
        rows.append((m, len(reps), sfull["d_min_sq"], sfull["multiplicity"],
                     str(sfull["mean_sq_dist"]), full_evals, orbit_evals,
                     eval_ratio, full_ms, orbit_ms, wall_speed, int(exact)))

    print(f"\n  C4 RESULT: {'PASS' if all_exact else 'FAIL'} "
          f"(orbit-reduced metric identical to full metric for every size)")
    print("  HEADLINE (exact, hardware-independent): the distance-evaluation count")
    print("  is reduced by EXACTLY 6x -- M(M-1) ordered pairs vs (M/6)(M-1).")
    print("  The 'wallspeed' column is corroboration only: it is memory/cache-bound")
    print("  and may sit below OR above 6x (the full M x M tally is more cache-")
    print("  hostile than the reduced (M/6) x M one). This is an OFFLINE")
    print("  constellation-design computation, not a per-symbol receive-path cost.")

    csv_path = os.path.join(args.results_dir, "sim03_symmetry.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "num_orbits", "d_min_sq", "multiplicity", "mean_sq_dist",
                    "full_evals", "orbit_evals", "eval_ratio",
                    "full_ms", "orbit_ms", "wall_speedup", "exact_match"])
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
