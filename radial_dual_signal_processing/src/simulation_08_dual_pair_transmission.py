"""
simulation_08_dual_pair_transmission.py - Study 8 (Episode III).

Backs claim C11: transmitting the inversion PAIR (x, iota_r(x)) -- an outer-zone
point and its mirrored inner-zone dual -- and decoding both with an exact integer
consistency cross-check gives a rate-1/2 block code whose structure is useful on
fading and impulsive channels, and whose consistency detector converts a
predictable majority of pair errors into DETECTED erasures. This is the first use
of inversion pairing on the channel itself (not in labels or storage).

Firewall: inversion builds the CODEBOOK; every receiver decision (joint ML, or
per-symbol ML plus the integer cross-check) is a true Euclidean nearest-point
decision on the actual received samples. Inversion never enters a metric.

Honest framing
--------------
On AWGN a rate-1/2 pair code buys nothing for free; the fair references are pure
repetition (send x twice) and the same average energy. The value, if any, is on
channels with independent per-symbol impairment (fading) and on impulsive
channels, where the exact consistency check (sectors must match; shells must
satisfy N * N_dual = r^4) flags impulse-struck pairs. Pair energies are unequal
by construction (N * N_dual = r^4 forces different totals across shell classes),
and self-dual points degenerate to pure repetition; both facts are computed
exactly up front and reported per shell class, never hidden.

Stage 1 (exact, no channel): product-distance table
---------------------------------------------------
The pair codebook's minimum 4D squared distance
  d2min_pair = min over distinct messages of |x-x'|^2 + |iota(x)-iota(x')|^2
is computed exactly (rational), together with the per-shell-class pair energies
and the self-dual (repetition-degenerate) message count. These exact numbers
pre-register the channel expectations BEFORE any Monte-Carlo run.

Stage 2 (channel): the two receivers vs baselines
-------------------------------------------------
  * joint 4D ML over the pair codebook (the performance ceiling);
  * per-symbol ML + exact integer consistency check (detects/erases inconsistent
    pairs), reporting detection rate and undetected-error (miss) rate;
  * baselines: pure repetition (x, x) with 4D ML, and -- to isolate the erasure
    CRITERION itself -- a generic soft-reliability detector that erases exactly
    as many pairs as the integer check at every operating point, choosing the
    pairs with the smallest joint-ML margin (best minus second-best 4D
    distance). Both criteria are scored with the SAME joint-ML estimator and a
    matched erasure count, so any gap is attributable to the criterion alone;
    the margin rule receives an oracle-tuned threshold for free, the most
    generous version of that competitor.
Additionally, Stage 1 verifies the isometric-leg theorem numerically: a
rotated-repetition codebook (x, R x) with R the order-6 lattice rotation has a
4D distance spectrum IDENTICAL to repetition's, because any isometric second
leg preserves per-leg distances. The 8x nearest-neighbor thinning of the
inversion pair is therefore achievable only because iota_r is NOT an isometry
-- the converse face of the firewall.
All CRN-paired with exact Clopper-Pearson intervals; AWGN, Rayleigh (independent
per-symbol fades), and impulsive channels.

Outputs
-------
  sim08_product_distance.csv   exact d2min_pair, energies, self-dual count,
                               rotated-repetition (isometry) check
  sim08_awgn.csv / _rayleigh.csv / _impulsive.csv   per-Es/N0 SER, detector
                               rates, and the criterion comparison at matched
                               erasure count (integer vs margin)
  sim08_summary.csv            headline comparisons per channel
  sim08_provenance.json

Reproduce: python simulation_08_dual_pair_transmission.py

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

import tqf_hex_signal as h
import tqf_admissibility as adm

SEED = 42
R_SQ = 12                       # inversion radius^2 (the M=54 radial dual object)
MAX_NORM = 144
ESN0_GRID_DB = list(range(0, 25))
TRIALS = 100_000
ALPHA = 0.05
IMPULSE_P = 0.05
IMPULSE_AMP = 5.0


def _build_pair_codebook(r_sq: int, max_norm: int):
    """Build the dual-pair codebook from the radial dual constellation.

    Each message m is a constellation point x_m; the transmitted pair is
    (x_m, iota_r(x_m)) where the inversion dual is another constellation point
    (the object is inversion-paired). Returns:
      x    : (M,) complex outer/primary points (unit-energy constellation)
      xinv : (M,) complex inversion-dual points (also constellation points)
      dual_index : (M,) index of each point's inversion dual
      self_dual  : (M,) bool, True where the point is its own dual (boundary shell)
      shell_of   : (M,) integer shell norm per point
    """
    con = h.build_radial_dual_constellation(r_sq=r_sq, max_norm_sq=max_norm)
    assert con.inversion_paired, "codebook requires an inversion-paired constellation"
    x = con.points_unit
    di = con.inversion_dual_index
    xinv = x[di]
    self_dual = (di == np.arange(con.size))
    shell_of = np.array([adm.shell_norm_sq(int(a), int(b)) for a, b in con.ab],
                        dtype=np.int64)
    return con, x, xinv, di, self_dual, shell_of


def _exact_product_distance(con, r_sq: int) -> Dict[str, object]:
    """Exact minimum 4D squared distance of the pair codebook and per-class
    energies, all rational. |x|^2 in unit-energy terms is scale_sq * norm; the
    pair energy of message m is scale_sq * (N_m + N_dual(m))."""
    ab = con.ab
    di = con.inversion_dual_index
    scale_sq = con.scale_sq_exact
    M = con.size
    norms = [adm.shell_norm_sq(int(a), int(b)) for a, b in ab]

    # exact squared distance in lattice units between points i and j:
    # ||p_i - p_j||^2 = da^2 + da*db + db^2 (integer); scale by scale_sq for energy.
    def latt_d2(i, j):
        da = int(ab[i, 0] - ab[j, 0]); db = int(ab[i, 1] - ab[j, 1])
        return da * da + da * db + db * db

    best = None
    mult = 0
    rep_best = None
    rep_mult = 0
    rot_best = None
    rot_mult = 0
    # rotated-repetition baseline: second leg = R(x), R the order-6 lattice
    # rotation (a, b) -> (-b, a+b). R is an ISOMETRY, so this codebook's 4D
    # spectrum must equal repetition's exactly -- |R x_i - R x_j| = |x_i - x_j|
    # for every pair, hence 4D distance 2 * d2(i, j) identically. The check
    # below verifies the theorem numerically; its content is that NO isometric
    # second-leg map can thin the joint spectrum, so the 8x multiplicity
    # reduction is achievable only because inversion is NOT an isometry.
    rot_of = {}
    coord_to_idx = {(int(a), int(b)): k for k, (a, b) in enumerate(ab)}
    for k in range(M):
        ra, rb = -int(ab[k, 1]), int(ab[k, 0]) + int(ab[k, 1])
        rot_of[k] = coord_to_idx.get((ra, rb), None)
    rotation_closed = all(v is not None for v in rot_of.values())

    for i in range(M):
        ii = int(di[i])
        for j in range(M):
            if i == j:
                continue
            jj = int(di[j])
            # 4D squared distance (lattice units): primary leg + dual leg
            d2 = latt_d2(i, j) + latt_d2(ii, jj)
            if best is None or d2 < best:
                best, mult = d2, 1
            elif d2 == best:
                mult += 1
            # repetition codebook (x, x): 4D distance is 2 * single-leg distance
            dr = 2 * latt_d2(i, j)
            if rep_best is None or dr < rep_best:
                rep_best, rep_mult = dr, 1
            elif dr == rep_best:
                rep_mult += 1
            # rotated repetition (x, R x): isometric second leg
            if rotation_closed:
                dv = latt_d2(i, j) + latt_d2(rot_of[i], rot_of[j])
                if rot_best is None or dv < rot_best:
                    rot_best, rot_mult = dv, 1
                elif dv == rot_best:
                    rot_mult += 1
    if rotation_closed:
        assert rot_best == rep_best and rot_mult == rep_mult, (
            "isometry theorem violated: rotated-repetition spectrum must equal "
            "repetition's")
    d2min_pair_energy = Fraction(best) * scale_sq   # exact energy-normalized

    # per shell-class pair energies (exact rational, unit-energy normalized)
    pair_energy = {}
    for i in range(M):
        N = norms[i]; Nd = norms[int(di[i])]
        pe = (Fraction(N) + Fraction(Nd)) * scale_sq
        pair_energy.setdefault((min(N, Nd), max(N, Nd)), pe)

    self_dual_count = int((di == np.arange(M)).sum())

    return {
        "M": M, "r_sq": r_sq,
        "d2min_pair_lattice": best,
        "d2min_pair_energy": str(d2min_pair_energy),
        "d2min_pair_energy_f": float(d2min_pair_energy),
        "d2min_repetition_lattice": rep_best,
        "pair_nn_multiplicity": mult,
        "repetition_nn_multiplicity": rep_mult,
        "multiplicity_ratio_rep_over_pair": rep_mult / mult,
        "rotrep_d2min_lattice": rot_best if rotation_closed else -1,
        "rotrep_nn_multiplicity": rot_mult if rotation_closed else -1,
        "isometric_leg_equals_repetition": int(
            rotation_closed and rot_best == rep_best and rot_mult == rep_mult),
        "self_dual_count": self_dual_count,
        "pair_energy_classes": {f"{k[0]}x{k[1]}": str(v) for k, v in pair_energy.items()},
        "scale_sq": str(scale_sq),
    }


def _ci(errors: int, trials: int) -> Tuple[float, float, float]:
    lo, hi = h.clopper_pearson(errors, trials, ALPHA)
    return errors / trials, lo, hi


def _per_symbol_with_check(rx1, rx2, con, x, xinv, di):
    """Per-symbol ML on each leg, then the exact integer consistency check.

    Decode leg 1 to nearest constellation point a, leg 2 to nearest b. The pair is
    CONSISTENT iff b is the inversion dual of a (di[a] == b). Consistent pairs
    report message a; inconsistent pairs are DETECTED errors (erasures). Returns
    (message_est, detected_mask) where detected_mask marks inconsistent (erased)
    pairs.
    """
    a = h.decode_ml(rx1, x)
    b = h.decode_ml(rx2, x)
    consistent = (di[a] == b)
    est = a.copy()
    detected = ~consistent
    return est, detected


def _run_channel(channel: str, con, x, xinv, di, self_dual,
                 rng: np.random.Generator) -> List[dict]:
    """Sweep one channel. All randomness is drawn explicitly, once per Es/N0
    point: the two legs of a codeword get INDEPENDENT draws (noise, impulse
    hits, fades), and the pair scheme and the repetition baseline share those
    per-leg draws (common random numbers ACROSS SCHEMES, never across the legs
    of one codeword)."""
    M = con.size
    rows = []
    for esn0 in ESN0_GRID_DB:
        tx = rng.integers(0, M, size=TRIALS)
        s1 = x[tx]
        s2 = xinv[tx]

        sigma_sq = h._noise_sigma_sq(esn0, 1.0)
        n1 = h._complex_gaussian(TRIALS, sigma_sq, rng)   # leg-1 noise
        n2 = h._complex_gaussian(TRIALS, sigma_sq, rng)   # leg-2 noise, independent

        if channel == "awgn":
            r1, r2 = s1 + n1, s2 + n2
            rr1, rr2 = x[tx] + n1, x[tx] + n2             # repetition, same legs
        elif channel == "rayleigh":
            h1 = h._complex_gaussian(TRIALS, 1.0, rng)    # leg-1 fade
            h2 = h._complex_gaussian(TRIALS, 1.0, rng)    # leg-2 fade, independent
            r1 = (h1 * s1 + n1) / h1                      # perfect-CSI zero forcing
            r2 = (h2 * s2 + n2) / h2
            rr1 = (h1 * x[tx] + n1) / h1
            rr2 = (h2 * x[tx] + n2) / h2
        elif channel == "impulsive":
            hit1 = rng.random(TRIALS) < IMPULSE_P
            hit2 = rng.random(TRIALS) < IMPULSE_P          # independent hits per leg
            th1 = rng.uniform(0.0, 2.0 * math.pi, TRIALS)
            th2 = rng.uniform(0.0, 2.0 * math.pi, TRIALS)
            imp1 = np.where(hit1, IMPULSE_AMP * np.exp(1j * th1), 0.0)
            imp2 = np.where(hit2, IMPULSE_AMP * np.exp(1j * th2), 0.0)
            r1, r2 = s1 + n1 + imp1, s2 + n2 + imp2
            rr1, rr2 = x[tx] + n1 + imp1, x[tx] + n2 + imp2
        else:
            raise ValueError(channel)

        # joint 4D ML (keep the full distance matrix for the margin baseline)
        d_pair = (np.abs(r1[:, None] - x[None, :]) ** 2
                  + np.abs(r2[:, None] - xinv[None, :]) ** 2)
        est_joint = np.argmin(d_pair, axis=1)
        err_joint = int((est_joint != tx).sum())

        # per-symbol + consistency check (the cheap O(1) receiver: leg-1 estimate)
        est_chk, detected = _per_symbol_with_check(r1, r2, con, x, xinv, di)
        # undetected error: pair passed the check but message is wrong
        undetected = (~detected) & (est_chk != tx)
        err_chk_undetected = int(undetected.sum())
        n_detected = int(detected.sum())
        # of the detected (erased) pairs, how many were actually wrong (true positives)
        would_be_wrong = (est_chk != tx)
        detected_true = int((detected & would_be_wrong).sum())

        # --- erasure-CRITERION comparison at matched erasure count ---------
        # Both detectors below use the SAME estimator (joint 4D ML) so the
        # comparison isolates the erasure criterion itself.
        # (a) integer criterion + joint estimator:
        intjoint_undetected = int(((~detected) & (est_joint != tx)).sum())
        # (b) generic soft-reliability criterion + joint estimator: erase the
        # n_detected pairs with the SMALLEST joint-ML margin d2 - d1 (best vs
        # second-best 4D distance). The erasure count is matched to the integer
        # detector at every operating point -- the most generous version of the
        # baseline, since it gets an oracle-tuned threshold for free.
        part = np.partition(d_pair, 1, axis=1)
        margin = part[:, 1] - part[:, 0]
        if n_detected > 0:
            thresh_idx = np.argpartition(margin, n_detected - 1)[:n_detected]
            margin_erased = np.zeros(TRIALS, dtype=bool)
            margin_erased[thresh_idx] = True
        else:
            margin_erased = np.zeros(TRIALS, dtype=bool)
        margin_undetected = int(((~margin_erased) & (est_joint != tx)).sum())

        # baseline: pure repetition (x, x) with joint ML, on the SAME per-leg
        # draws as the pair scheme (rr1/rr2 built above)
        d_rep = np.abs(rr1[:, None] - x[None, :]) ** 2 + np.abs(rr2[:, None] - x[None, :]) ** 2
        est_rep = np.argmin(d_rep, axis=1)
        err_rep = int((est_rep != tx).sum())

        sj, sj_lo, sj_hi = _ci(err_joint, TRIALS)
        sc, sc_lo, sc_hi = _ci(err_chk_undetected, TRIALS)
        sr, sr_lo, sr_hi = _ci(err_rep, TRIALS)
        si, si_lo, si_hi = _ci(intjoint_undetected, TRIALS)
        sm, sm_lo, sm_hi = _ci(margin_undetected, TRIALS)

        rows.append({
            "channel": channel, "esn0_db": esn0, "trials": TRIALS,
            "joint_ser": sj, "joint_ser_lo": sj_lo, "joint_ser_hi": sj_hi,
            "check_undetected_ser": sc, "check_undet_lo": sc_lo, "check_undet_hi": sc_hi,
            "repetition_ser": sr, "rep_lo": sr_lo, "rep_hi": sr_hi,
            "detected_fraction": n_detected / TRIALS,
            "detected_true_positive": detected_true,
            "detected_total": n_detected,
            "joint_vs_rep": sj / sr if sr > 0 else float("nan"),
            "intjoint_undetected_ser": si,
            "intjoint_undet_lo": si_lo, "intjoint_undet_hi": si_hi,
            "margin_undetected_ser": sm,
            "margin_undet_lo": sm_lo, "margin_undet_hi": sm_hi,
            "margin_erased_matched": int(n_detected),
            "intcheck_vs_margin": (si / sm) if sm > 0 else float("nan"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=str, default=".")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    con, x, xinv, di, self_dual, shell_of = _build_pair_codebook(R_SQ, MAX_NORM)

    # Stage 1: exact product-distance table (freezes expectations).
    pd = _exact_product_distance(con, R_SQ)

    rng = np.random.default_rng(SEED)
    awgn_rows = _run_channel("awgn", con, x, xinv, di, self_dual, rng)
    rayl_rows = _run_channel("rayleigh", con, x, xinv, di, self_dual, rng)
    imp_rows = _run_channel("impulsive", con, x, xinv, di, self_dual, rng)

    def _write(name, rows):
        with open(os.path.join(args.results_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # product-distance table (flatten the per-class dict)
    pd_row = {k: v for k, v in pd.items() if k != "pair_energy_classes"}
    pd_row["pair_energy_classes"] = ";".join(
        f"{k}={v}" for k, v in pd["pair_energy_classes"].items())
    _write("sim08_product_distance.csv", [pd_row])
    _write("sim08_awgn.csv", awgn_rows)
    _write("sim08_rayleigh.csv", rayl_rows)
    _write("sim08_impulsive.csv", imp_rows)

    # summary: detector effectiveness on impulsive; joint-vs-rep on rayleigh
    def _at(rows, esn0):
        for r in rows:
            if r["esn0_db"] == esn0:
                return r
        return rows[-1]
    summary = [
        {"channel": "impulsive", "esn0_db": 10,
         "detected_fraction": _at(imp_rows, 10)["detected_fraction"],
         "check_undetected_ser": _at(imp_rows, 10)["check_undetected_ser"],
         "joint_ser": _at(imp_rows, 10)["joint_ser"]},
        {"channel": "rayleigh", "esn0_db": 15,
         "detected_fraction": _at(rayl_rows, 15)["detected_fraction"],
         "check_undetected_ser": _at(rayl_rows, 15)["check_undetected_ser"],
         "joint_ser": _at(rayl_rows, 15)["joint_ser"]},
        {"channel": "awgn", "esn0_db": 15,
         "detected_fraction": _at(awgn_rows, 15)["detected_fraction"],
         "check_undetected_ser": _at(awgn_rows, 15)["check_undetected_ser"],
         "joint_ser": _at(awgn_rows, 15)["joint_ser"]},
    ]
    _write("sim08_summary.csv", summary)

    provenance = {
        "study": 8, "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "tqf_hex_signal_version": h.__version__,
        "r_sq": R_SQ, "max_norm": MAX_NORM, "M": con.size,
        "esn0_grid_db": ESN0_GRID_DB, "trials": TRIALS, "alpha": ALPHA,
        "impulse_p": IMPULSE_P, "impulse_amp": IMPULSE_AMP,
    }
    with open(os.path.join(args.results_dir, "sim08_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("Study 8 complete.")
    print(f"  codebook M={con.size} (r_sq={R_SQ}), self-dual (repetition) msgs="
          f"{pd['self_dual_count']}")
    print(f"  exact d2min_pair (energy) = {pd['d2min_pair_energy_f']:.4f}  "
          f"[pair d2min == repetition d2min: "
          f"{pd['d2min_pair_lattice'] == pd['d2min_repetition_lattice']}; "
          f"NN multiplicity pair={pd['pair_nn_multiplicity']} vs "
          f"rep={pd['repetition_nn_multiplicity']} "
          f"({pd['multiplicity_ratio_rep_over_pair']:.1f}x)]")
    imp10 = _at(imp_rows, 10)
    print(f"  isometric-leg check (rotated repetition == repetition): "
          f"{bool(pd['isometric_leg_equals_repetition'])}")
    print(f"  impulsive @10dB: detected {imp10['detected_fraction']*100:.1f}% of pairs, "
          f"undetected SER {imp10['check_undetected_ser']:.2e}, joint SER {imp10['joint_ser']:.2e}")
    imp20 = _at(imp_rows, 20)
    print(f"  criterion @20dB impulsive (matched erasures, joint estimator): "
          f"integer {imp20['intjoint_undetected_ser']:.2e} vs "
          f"margin {imp20['margin_undetected_ser']:.2e} "
          f"(int/margin {imp20['intcheck_vs_margin']:.2f})")
    awgn20 = _at(awgn_rows, 20)
    print(f"  criterion @20dB awgn: integer {awgn20['intjoint_undetected_ser']:.2e} vs "
          f"margin {awgn20['margin_undetected_ser']:.2e} "
          f"(int/margin {awgn20['intcheck_vs_margin']:.2f})")
    ray15 = _at(rayl_rows, 15)
    print(f"  rayleigh @15dB: joint SER {ray15['joint_ser']:.2e}, "
          f"repetition SER {ray15['repetition_ser']:.2e}, joint/rep {ray15['joint_vs_rep']:.2f}")


if __name__ == "__main__":
    main()
