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

Output: the results table (sizes, speedup, exact-match flags).

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.1.0
Date: June 27, 2026
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


def _combined_orbit_partition(con: t.Constellation) -> Tuple[List[int], List[int]]:
    """Partition the radial-dual constellation into orbits under (a) the order-6
    rotation C6 alone and (b) the order-12 rotation x inversion group C6 x Z2.

    Returns (rotation_reps, combined_reps) as lists of representative indices.
    Inversion is applied here as the exact same-sector POINT PERMUTATION
    (con.inversion_dual_index) -- a label/structure operation, never a metric.
    """
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    coord_to_idx = {(int(a[i]), int(b[i])): i for i in range(con.size)}
    dual = np.asarray(con.inversion_dual_index, dtype=np.int64)

    # Rotation-only orbits.
    seen_rot = set()
    rot_reps: List[int] = []
    for i in range(con.size):
        if i in seen_rot:
            continue
        ca, cb = int(a[i]), int(b[i])
        for _ in range(6):
            seen_rot.add(coord_to_idx[(ca, cb)])
            ca, cb = t.rotate60(ca, cb)
        rot_reps.append(i)

    # Combined orbits: close each point under rotation AND the inversion dual.
    seen_all = set()
    comb_reps: List[int] = []
    for i in range(con.size):
        if i in seen_all:
            continue
        frontier = [i]
        orbit = set()
        while frontier:
            j = frontier.pop()
            if j in orbit:
                continue
            orbit.add(j)
            ca, cb = int(a[j]), int(b[j])
            rj = coord_to_idx[t.rotate60(ca, cb)]      # rotation neighbour
            if rj not in orbit:
                frontier.append(rj)
            ij = int(dual[j])                          # inversion neighbour
            if ij not in orbit:
                frontier.append(ij)
        seen_all |= orbit
        comb_reps.append(i)
    return rot_reps, comb_reps


def _shell_incidence(con: t.Constellation, max_norm_sq: int) -> np.ndarray:
    """Discrete, inversion-invariant LABEL enumerator: points per shell norm."""
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    norms = a * a + a * b + b * b
    return np.bincount(norms, minlength=max_norm_sq + 1).astype(np.int64)


def radial_dual_reduction_block(r_sq: int, max_norm_sq: int,
                                results_dir: str) -> bool:
    """Combined rotation + inversion symmetry reduction on the C7 radial-dual
    constellation, with the firewall stated explicitly.

    Two clearly separated reductions, NOT multiplied:
      * EUCLIDEAN (the C4 metric): the pairwise squared-distance enumerator folds
        by rotation EXACTLY 6x and no more. Inversion is conformal, not isometric,
        so it does NOT reduce a Euclidean-distance computation -- verified here by
        showing the inversion point-permutation does not preserve per-pair squared
        distances even though it is a constellation symmetry.
      * LABEL/STRUCTURE: a discrete inversion-invariant enumerator (points per
        shell) and the orbit structure fold by the FULL order-12 group C6 x Z2,
        giving fewer combined orbits than rotation alone. This is a storage/label
        statement about the constellation, kept distinct from the metric.
    """
    con = t.build_radial_dual_constellation(r_sq, max_norm_sq)
    m = con.size
    size = int(4 * max(int(n) for n in con.shell_norms) + 1)

    # (A) EUCLIDEAN squared-distance enumerator: full vs rotation fold (exact 6x).
    rot_reps, comb_reps = _combined_orbit_partition(con)
    reps = np.array(rot_reps, dtype=np.int64)
    enum_full = _pair_enumerator_full(con.ab, size)
    enum_rot = _pair_enumerator_orbit(con.ab, reps, size)
    euclid_exact = bool(np.array_equal(enum_full, enum_rot))
    euclid_ratio = (m * (m - 1)) / (reps.shape[0] * (m - 1))   # == 6.0

    # Firewall demonstration: inversion does NOT preserve per-pair squared
    # distances (so it cannot fold the Euclidean enumerator), even though it is a
    # constellation symmetry. Compare squared distances from point 0 vs its dual.
    dual = np.asarray(con.inversion_dual_index, dtype=np.int64)
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    da0 = a - a[0]; db0 = b - b[0]
    d0 = da0 * da0 + da0 * db0 + db0 * db0          # squared dists from point 0
    ai = a[dual]; bi = b[dual]
    dai = ai - ai[0]; dbi = bi - bi[0]
    di = dai * dai + dai * dbi + dbi * dbi          # squared dists from dual(0)
    inversion_is_isometry = bool(np.array_equal(np.sort(d0), np.sort(di)))

    # (B) LABEL enumerator: points per shell, full vs rotation vs combined fold.
    incid_full = _shell_incidence(con, max_norm_sq)
    # rotation fold: one rep per shell -> x6 reproduces the incidence exactly.
    incid_rot = np.zeros_like(incid_full)
    for i in rot_reps:
        n = int(a[i] * a[i] + a[i] * b[i] + b[i] * b[i])
        incid_rot[n] += 6
    # combined fold: store inner/boundary orbit reps; regenerate outer via dual.
    incid_comb = np.zeros_like(incid_full)
    for i in comb_reps:
        ni = int(a[i] * a[i] + a[i] * b[i] + b[i] * b[i])
        di_idx = int(dual[i])
        nd = int(a[di_idx] * a[di_idx] + a[di_idx] * b[di_idx] + b[di_idx] * b[di_idx])
        incid_comb[ni] += 6
        if nd != ni:                                # not self-dual -> add the dual shell
            incid_comb[nd] += 6
    label_exact = bool(np.array_equal(incid_full, incid_rot) and
                       np.array_equal(incid_full, incid_comb))
    label_ratio_rot = m / len(rot_reps)             # 6x on the label enumerator
    label_ratio_comb = m / len(comb_reps)           # >6x (combined)

    print("\n" + "-" * 84)
    print("  C4 + C7 combined-symmetry reduction on the radial-dual constellation")
    print(f"  (M={m}, shells={list(con.shell_norms)}, r^2={r_sq})")
    print("-" * 84)
    print(f"  [EUCLIDEAN metric] squared-distance enumerator: full vs rotation fold")
    print(f"    exact integer match: {euclid_exact}; evaluation reduction = "
          f"{euclid_ratio:.3f}x (rotation only, EXACTLY 6x).")
    print(f"    inversion is an isometry on this object: {inversion_is_isometry} "
          f"-> inversion does NOT fold the Euclidean enumerator (firewall).")
    print(f"  [LABEL/structure] points-per-shell enumerator: full vs rotation vs "
          f"combined")
    print(f"    exact integer match (all three): {label_exact}")
    print(f"    rotation orbits = {len(rot_reps)} ({label_ratio_rot:.3f}x); "
          f"combined C6 x Z2 orbits = {len(comb_reps)} ({label_ratio_comb:.3f}x).")
    print("    The combined (>6x) factor is a LABEL/STORAGE reduction on a discrete")
    print("    inversion-invariant enumerator -- it is NOT multiplied with, nor a")
    print("    substitute for, the exact 6x Euclidean metric reduction above.")
    block_pass = euclid_exact and label_exact and (not inversion_is_isometry) and (
        abs(euclid_ratio - 6.0) < 1e-9)
    print(f"  COMBINED-BLOCK RESULT: {'PASS' if block_pass else 'FAIL'}")

    out_path = os.path.join(results_dir, "sim03_inversion_reduction.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "domain", "full", "rotation_fold",
                    "combined_fold", "reduction_rotation", "reduction_combined",
                    "exact_match", "inversion_is_isometry"])
        w.writerow(["sq_distance_enumerator", "euclidean", m * (m - 1),
                    reps.shape[0] * (m - 1), "", euclid_ratio, "",
                    int(euclid_exact), int(inversion_is_isometry)])
        w.writerow(["shell_incidence", "label", m, len(rot_reps), len(comb_reps),
                    label_ratio_rot, label_ratio_comb, int(label_exact),
                    int(inversion_is_isometry)])
    print(f"  Wrote {out_path}")
    return block_pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max_norm_sq", type=int, nargs="+",
                    default=[7, 19, 37, 61, 91, 127],
                    help="squared-norm thresholds defining 6-fold-symmetric "
                         "disk constellations of increasing size")
    ap.add_argument("--timing_repeats", type=int, default=11)
    ap.add_argument("--seed", type=int, default=42)  # determinism only; unused
    ap.add_argument("--rd_r_sq", type=int, default=12,
                    help="radial-dual inversion radius^2 for the combined-symmetry "
                         "block (C7 object)")
    ap.add_argument("--rd_max_norm_sq", type=int, default=60,
                    help="largest shell norm for the radial-dual combined-symmetry "
                         "block")
    ap.add_argument("--skip_inversion_block", action="store_true",
                    help="skip the combined rotation+inversion block (kept additive; "
                         "existing sim03_symmetry.csv is unaffected either way)")
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

    # Additive combined rotation + inversion block on the C7 radial-dual
    # constellation. The disk-constellation results and sim03_symmetry.csv above
    # are unaffected; this writes a separate sim03_inversion_reduction.csv.
    if not args.skip_inversion_block:
        radial_dual_reduction_block(args.rd_r_sq, args.rd_max_norm_sq,
                                    args.results_dir)
    print("=" * 84)


if __name__ == "__main__":
    main()
