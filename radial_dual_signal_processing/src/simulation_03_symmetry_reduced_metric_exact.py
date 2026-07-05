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

Exact constellation-geometry / fairness block
---------------------------------------------
A block (``--skip_geometry_block`` to disable) applies the same exact
integer/rational machinery to the C3 comparison constellations themselves. For
hexagonal C_M and square QAM at M in {16, 64, 256} -- plus the C7 radial-dual
constellation and its matched filled hex-42 baseline -- it computes EXACTLY
(fractions.Fraction, self-checked with ``==`` against a pre-registered table):

  * d_min^2 at unit average energy, and the pure-d_min^2 predicted gain
    10*log10(d2_hex / d2_sq);
  * the average nearest-neighbor multiplicity K_bar (ordered NN pairs / M);
  * the peak-to-average power ratio (PAPR) -- average-energy matching hides no
    peak-power penalty, and hex's PAPR is <= square's at every M;
  * the labeling quality: mean nearest-neighbor Hamming distance (square's
    true Gray map is exactly 1; the hex Gray-like map is 1.73-2.43, which
    PREDICTS the BER-vs-SER inversion instead of merely exhibiting it);
  * a nearest-neighbor-approximation prediction, SER ~= K_bar * Q(d_min /
    (sigma*sqrt(2))) with the same complex-noise convention as the channel
    code, solved for the Eb/N0 that reaches each target SER. The predicted
    hex-vs-square gains quantitatively explain the measured C3 gains: the
    finite-M d_min^2 gain is 0.46-0.81 dB (ABOVE the 0.6 dB asymptote for
    M >= 64); the larger hex multiplicity is what pulls the net measured gain
    down to 0.29-0.50 dB.

No randomness is used and the pre-existing sim03 CSVs are byte-identical; the
block writes two new CSVs (sim03_constellation_geometry.csv and
sim03_nn_prediction.csv).

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
            rj = coord_to_idx[t.rotate60(ca, cb)]      # rotation neighbor
            if rj not in orbit:
                frontier.append(rj)
            ij = int(dual[j])                          # inversion neighbor
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


# ---------------------------------------------------------------------------
# Exact constellation-geometry / fairness block
# ---------------------------------------------------------------------------

def _exact_lattice_geometry(con: t.Constellation
                            ) -> Tuple[Fraction, int, Fraction, int]:
    """Exact (d_min^2, ordered NN-pair count, PAPR, lattice d_min^2) for a
    lattice constellation, at its unit-average-energy normalization.

    All pairwise squared distances are the integer Eisenstein norms of the
    coordinate differences, so d_min^2 = (min lattice norm) * scale_sq_exact and
    PAPR = (max point norm) * scale_sq_exact are exact rationals.
    """
    ab = con.ab.astype(np.int64)
    a = ab[:, 0]; b = ab[:, 1]
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    sq = da * da + da * db + db * db
    np.fill_diagonal(sq, np.iinfo(np.int64).max)
    d_lat = int(sq.min())
    nn_pairs = int(np.sum(sq == d_lat))                 # ordered pairs
    peak = int(np.max(a * a + a * b + b * b))
    s2 = con.scale_sq_exact
    return Fraction(d_lat) * s2, nn_pairs, Fraction(peak) * s2, d_lat


def _hex_gray_nn_hamming(con: t.Constellation, d_lat_min: int) -> Fraction:
    """Exact mean Hamming distance over ordered nearest-neighbor label pairs."""
    ab = con.ab.astype(np.int64)
    a = ab[:, 0]; b = ab[:, 1]
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    sq = da * da + da * db + db * db
    np.fill_diagonal(sq, np.iinfo(np.int64).max)
    nn = np.argwhere(sq == d_lat_min)
    lab = con.labels
    total = sum(bin(int(lab[i]) ^ int(lab[j])).count("1") for i, j in nn)
    return Fraction(total, len(nn))


def _exact_square_geometry(m: int) -> Tuple[Fraction, int, Fraction, Fraction]:
    """Exact (d_min^2, ordered NN pairs, PAPR, mean NN Hamming) for square M-QAM.

    Per-axis amplitudes are the odd integers; the exact average energy is
    2*sum(amps^2)/side, the minimum step is 2, the peak is 2*(side-1)^2, the
    grid has 4*side*(side-1) ordered nearest-neighbor pairs, and the true
    per-axis Gray map makes every NN label pair differ in EXACTLY one bit.
    """
    side = int(round(math.sqrt(m)))
    assert side * side == m, "square QAM geometry requires a perfect square M"
    amps = [2 * i - (side - 1) for i in range(side)]
    avg = Fraction(2 * sum(x * x for x in amps), side)
    d2 = Fraction(4) / avg
    papr = Fraction(2 * (side - 1) ** 2) / avg
    nn_pairs = 4 * side * (side - 1)
    # true Gray: verify (not assume) the exact mean-NN-Hamming of 1 on the grid.
    con = t.build_square_qam(m)
    lab = con.labels.reshape(side, side)                # [i, q] layout by builder
    total = 0
    for i in range(side):
        for q in range(side):
            if i + 1 < side:
                total += 2 * bin(int(lab[i, q]) ^ int(lab[i + 1, q])).count("1")
            if q + 1 < side:
                total += 2 * bin(int(lab[i, q]) ^ int(lab[i, q + 1])).count("1")
    gray = Fraction(total, nn_pairs)
    return d2, nn_pairs, papr, gray


def _qfunc(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _nn_predicted_ebn0(d2: float, kbar: float, bits: float,
                       target: float) -> float:
    """Eb/N0 (dB) at which the nearest-neighbor approximation
    SER ~= K_bar * Q(d_min / (sigma * sqrt(2))) reaches ``target``.

    Convention matches the channel code exactly: unit Es, sigma^2 = N0 =
    1 / (Es/N0)_lin with Es/N0 (dB) = Eb/N0 (dB) + 10*log10(bits); complex
    CN(0, sigma^2) noise projects onto the line between two points as a real
    N(0, sigma^2 / 2), so the pairwise error is Q((d/2) / sqrt(sigma^2 / 2)).
    Solved by bisection (SER is monotone decreasing in Eb/N0).
    """
    lo, hi = -10.0, 80.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        sigma_sq = t._noise_sigma_sq(mid, bits)
        ser = kbar * _qfunc(math.sqrt(d2) / math.sqrt(2.0 * sigma_sq))
        if ser > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Pre-registered exact values (MARK3_PLAN T3-14, verified independently before
# implementation). The block FAILS if the code does not reproduce these with ==.
_GEOM_EXPECTED = {
    "hex-16":     dict(d2=Fraction(4, 9),      nn=66,   papr=Fraction(16, 9),
                       gray=Fraction(19, 11)),
    "hex-64":     dict(d2=Fraction(64, 567),   nn=326,  papr=Fraction(1216, 567),
                       gray=Fraction(342, 163)),
    "hex-256":    dict(d2=Fraction(256, 9027), nn=1422, papr=Fraction(18688, 9027),
                       gray=Fraction(1726, 711)),
    "sqQAM-16":   dict(d2=Fraction(2, 5),      nn=48,   papr=Fraction(9, 5),
                       gray=Fraction(1)),
    "sqQAM-64":   dict(d2=Fraction(2, 21),     nn=224,  papr=Fraction(7, 3),
                       gray=Fraction(1)),
    "sqQAM-256":  dict(d2=Fraction(2, 85),     nn=960,  papr=Fraction(45, 17),
                       gray=Fraction(1)),
    "radial-dual-r12-Nle60": dict(d2=Fraction(7, 128), nn=48,
                                  papr=Fraction(21, 8), gray=None),
    "hex-any-42": dict(d2=Fraction(7, 41),     nn=200,  papr=Fraction(84, 41),
                       gray=None),
}


def constellation_geometry_block(results_dir: str, targets=(1e-2, 1e-3),
                                 rd_r_sq: int = 12, rd_max_norm_sq: int = 60
                                 ) -> bool:
    """Exact geometry / fairness facts for the C3 constellations (+ the C7 pair).

    Everything here is a constellation PROPERTY -- no channel, no randomness --
    computed in exact rational arithmetic and checked with ``==`` against the
    pre-registered table above, then written to two CSVs. The NN-approximation
    predictions are the only floats (a Q-function has no exact form) and are the
    quantitative bridge from these exact facts to the measured Study 2 gains.
    """
    print("\n" + "-" * 84)
    print("  Mark 3: exact constellation geometry & fairness facts "
          "(d_min^2, K_bar, PAPR, Gray)")
    print("-" * 84)

    rows = []      # per-constellation facts
    facts = {}     # name -> dict for the prediction step
    hex_cons = {m: t.build_filled_constellation(m) for m in (16, 64, 256)}
    others = [t.build_radial_dual_constellation(rd_r_sq, rd_max_norm_sq),
              t.build_filled_constellation_any(42)]

    all_ok = True
    for con in list(hex_cons.values()) + others:
        d2, nn, papr, d_lat = _exact_lattice_geometry(con)
        gray = (_hex_gray_nn_hamming(con, d_lat)
                if con.name.startswith("hex-") and con.bits_per_symbol > 0
                else None)
        exp = _GEOM_EXPECTED[con.name]
        ok = (d2 == exp["d2"] and nn == exp["nn"] and papr == exp["papr"]
              and (exp["gray"] is None or gray == exp["gray"]))
        all_ok = all_ok and ok
        facts[con.name] = dict(d2=d2, kbar=Fraction(nn, con.size), M=con.size)
        rows.append((con.name, con.size, str(d2), float(d2), nn,
                     float(Fraction(nn, con.size)), str(papr), float(papr),
                     10.0 * math.log10(float(papr)),
                     (str(gray) if gray is not None else ""),
                     (float(gray) if gray is not None else float("nan")),
                     int(ok)))
    for m in (16, 64, 256):
        d2, nn, papr, gray = _exact_square_geometry(m)
        name = f"sqQAM-{m}"
        exp = _GEOM_EXPECTED[name]
        ok = (d2 == exp["d2"] and nn == exp["nn"] and papr == exp["papr"]
              and gray == exp["gray"] == Fraction(1))
        all_ok = all_ok and ok
        facts[name] = dict(d2=d2, kbar=Fraction(nn, m), M=m)
        rows.append((name, m, str(d2), float(d2), nn, float(Fraction(nn, m)),
                     str(papr), float(papr), 10.0 * math.log10(float(papr)),
                     str(gray), float(gray), int(ok)))

    print(f"  {'constellation':<24}{'M':>5}{'d_min^2':>14}{'K_bar':>8}"
          f"{'PAPR(dB)':>10}{'GrayNN':>8}{'==':>4}")
    for r in sorted(rows, key=lambda x: (x[1], x[0])):
        gtxt = f"{r[10]:.3f}" if r[10] == r[10] else "  -- "
        print(f"  {r[0]:<24}{r[1]:>5}{r[2]:>14}{r[5]:>8.3f}"
              f"{r[8]:>10.2f}{gtxt:>8}{('ok' if r[11] else 'FAIL'):>4}")
    print("  [PAPR fairness: hex <= square at every M -- matching AVERAGE energy")
    print("   hides no PEAK-power penalty. GrayNN: square's true Gray map is")
    print("   exactly 1 bit per NN step; the hex Gray-like map is 1.73-2.43,")
    print("   which predicts hex BER > square BER even where hex SER is lower.]")

    # NN-approximation predictions for the hex-vs-square pairs.
    pred_rows = []
    print(f"\n  NN-approx prediction SER ~= K_bar * Q(d_min/(sigma*sqrt(2))):")
    print(f"  {'M':>5}{'target':>9}{'Eb/N0 hex':>11}{'Eb/N0 sq':>10}"
          f"{'pred gain':>10}{'pure-dmin':>10}")
    for m in (16, 64, 256):
        h = facts[f"hex-{m}"]; q = facts[f"sqQAM-{m}"]
        bits = math.log2(m)
        pure = 10.0 * math.log10(float(h["d2"]) / float(q["d2"]))
        for target in targets:
            eh = _nn_predicted_ebn0(float(h["d2"]), float(h["kbar"]), bits, target)
            eq = _nn_predicted_ebn0(float(q["d2"]), float(q["kbar"]), bits, target)
            pred_rows.append((m, target, eh, eq, eq - eh, pure))
            print(f"  {m:>5}{target:>9.0e}{eh:>11.2f}{eq:>10.2f}"
                  f"{eq - eh:>+10.3f}{pure:>+10.3f}")
    print("  [The finite-M pure-d_min^2 gain is ABOVE the 0.6 dB asymptote for")
    print("   M >= 64; the larger hex NN multiplicity is what pulls the net,")
    print("   NN-predicted gain down to the 0.13-0.55 dB range that the measured")
    print("   Study 2 gains land in -- the exact two-factor account of C3.]")
    price = 10.0 * math.log10(float(facts["hex-any-42"]["d2"]) /
                              float(facts["radial-dual-r12-Nle60"]["d2"]))
    print(f"  C7 geometry-price preview (Study 8): pure-d_min^2 penalty of the")
    print(f"  radial-dual constellation vs filled hex-42 = {price:+.2f} dB "
          f"(7/128 vs 7/41).")

    print(f"\n  GEOMETRY-BLOCK RESULT: "
          f"{'PASS' if all_ok else 'FAIL'} (every exact value reproduced the "
          f"pre-registered table with ==)")

    geom_path = os.path.join(results_dir, "sim03_constellation_geometry.csv")
    with open(geom_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["constellation", "M", "d2min_exact", "d2min",
                    "ordered_nn_pairs", "avg_nn_multiplicity",
                    "papr_exact", "papr", "papr_db",
                    "gray_nn_hamming_exact", "gray_nn_hamming", "matches_expected"])
        w.writerows(rows)
    pred_path = os.path.join(results_dir, "sim03_nn_prediction.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "target_ser", "ebn0_hex_pred_db", "ebn0_sq_pred_db",
                    "gain_pred_db", "gain_pure_dmin_db"])
        w.writerows(pred_rows)
    print(f"  Wrote {geom_path}")
    print(f"  Wrote {pred_path}")
    return all_ok


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
    ap.add_argument("--skip_geometry_block", action="store_true",
                    help="skip the Mark 3 exact constellation-geometry / fairness "
                         "block (kept additive; existing sim03 CSVs are unaffected "
                         "either way)")
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

    # Exact-geometry / fairness block (self-checked with ==). No randomness;
    # the sim03 CSVs written above are unaffected.
    if not args.skip_geometry_block:
        constellation_geometry_block(args.results_dir,
                                     rd_r_sq=args.rd_r_sq,
                                     rd_max_norm_sq=args.rd_max_norm_sq)
    print("=" * 84)


if __name__ == "__main__":
    main()
