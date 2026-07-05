#!/usr/bin/env python3
"""
simulation_08_radial_dual_geometry_price.py - AWGN Geometry Price of the Radial-Dual Constellation (C7 honesty)

Measures what the C7 radial-dual constellation's duality structure COSTS in
plain AWGN error-rate geometry, for the Tri-Quarter Framework (TQF)
radial_dual_signal_processing subproject.

What this study is (and is not)
-------------------------------
The radial-dual constellation C_rd (M = 42, shells {3,4,9,12,16,36,48},
r^2 = 12) is a STRUCTURE demonstrator: it is the object on which the exact
rotation x inversion (C6 x Z2) folds, the bitwise-ML folded decoder, and the
order-12 differential codec live. It was never designed for AWGN symbol-error
performance, and this study states the expected verdict UP FRONT rather than
leaving the obvious reviewer question open: at matched order (M = 42) and
matched average energy, C_rd's AWGN SER is strictly WORSE than a plain filled
hexagonal constellation of the same size. This is a null-to-negative geometry
result for C_rd, reported as the measured PRICE of the duality structure --
it qualifies claim C7's scope; it introduces no new claim, and it does not
touch the exactness (C1), cost (C2), or structure (C7/C8) columns, which are
what C_rd actually buys.

Pre-registered expectation (exact, computed before this script ran)
-------------------------------------------------------------------
At unit average energy, in exact rational arithmetic:
  * C_rd:            d_min^2 = 7/128 ~= 0.0547,  PAPR = 21/8   (4.19 dB)
  * filled hex-42:   d_min^2 = 7/41  ~= 0.1707,  PAPR = 84/41  (3.11 dB)
so the pure-d_min^2 penalty is 10*log10((7/41)/(7/128)) = 10*log10(128/41)
~= +4.94 dB against C_rd. But the nearest-neighbor multiplicities differ
sharply (K_bar = 48/42 ~= 1.14 for C_rd vs 200/42 ~= 4.76 for hex-42), and
C_rd's LOW multiplicity is a large offset, not a small one: the nearest-
neighbor approximation SER ~= K_bar * Q(d_min/(sigma*sqrt(2))) -- the SAME
model that predicts the Study 2 hex-vs-square gains -- folds the +4.94 dB
d_min penalty together with the -6.2 dB prefactor (multiplicity) ratio into a
NET predicted Es/N0 price of ~+3.33 dB at SER 1e-2, rising to ~+3.9 dB at 1e-3
as the d_min term reasserts. The pre-registered bracket is therefore 2.8-4.0 dB
at SER 1e-2, CENTERED on that model prediction rather than on the raw d_min
penalty. The script computes the NN-approximation price internally and prints a
verdict against the bracket either way -- the expectation is a model
prediction, not a guess, and it is falsifiable.

Methodology (identical discipline to Study 2)
---------------------------------------------
* Matched order (both M = 42) and matched average energy (exact rational unit-
  energy scales); SER is the metric (both objects are label-free, so SER is
  the honest, labeling-independent comparison).
* Axis: Es/N0 in dB. Both constellations have bits_per_symbol = 0, so the
  channel helper's Eb/N0 argument clamps to 1 bit and parameterizes Es/N0
  directly. (With log2(42) bits attached, Eb/N0 = Es/N0 - 7.29 dB.)
* Common random numbers: the identical data-index stream and the identical
  complex-noise realization drive both constellations at every grid point
  (RNG-state rewind between the two channel calls), so per-point differences
  are attributable to the geometry, not the seed.
* Exact Clopper-Pearson intervals on every SER point; per-point significance
  by the exact McNemar test on the discordant pairs, Holm-corrected across the
  grid; the marginal CI-disjointness flag is retained as a companion.
* The Es/N0 gap at each target SER is read by the same log-linear
  interpolation as Study 2, with a conservative/optimistic band from the
  Clopper-Pearson curves.
* C1 tie-in on BOTH objects, for free: hex-42 is decoded by the closed-form
  O(1)-fast-path decoder and C_rd by the phase-pair + inversion FOLDED decoder,
  and each is asserted bitwise-identical to exhaustive ML at every grid point
  before its errors are counted. The inversion firewall holds throughout:
  every Euclidean decision is on true distances; inversion only regenerates
  stored labels inside the folded decoder.

Output: the exact-geometry preamble, the per-point SER table with paired
significance, the price-at-target readout with its pre-registration verdict,
and the C1 tie-in PASS/FAIL banner.

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
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

import tqf_hex_signal as t


# ---------------------------------------------------------------------------
# Exact geometry preamble (pre-registered; checked with ==)
# ---------------------------------------------------------------------------

def _exact_geometry(con: t.Constellation) -> Tuple[Fraction, int, Fraction]:
    """Exact (d_min^2, ordered NN-pair count, PAPR) at unit average energy."""
    ab = con.ab.astype(np.int64)
    a = ab[:, 0]; b = ab[:, 1]
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    sq = da * da + da * db + db * db
    np.fill_diagonal(sq, np.iinfo(np.int64).max)
    d_lat = int(sq.min())
    nn_pairs = int(np.sum(sq == d_lat))
    peak = int(np.max(a * a + a * b + b * b))
    s2 = con.scale_sq_exact
    return Fraction(d_lat) * s2, nn_pairs, Fraction(peak) * s2


# Pre-registered exact values (MARK3_PLAN T3-16 / sim03 geometry block).
_EXPECTED = {
    "radial-dual-r12-Nle60": dict(d2=Fraction(7, 128), nn=48,
                                  papr=Fraction(21, 8)),
    "hex-any-42":            dict(d2=Fraction(7, 41),  nn=200,
                                  papr=Fraction(84, 41)),
}
# Expected MEASURED Es/N0 gap @ SER 1e-2, CENTERED on the nearest-neighbor-
# approximation price (~+3.33 dB): the pure-d_min^2 penalty (+4.94 dB) folded
# with C_rd's much lower NN multiplicity (K_bar 1.14 vs 4.76, a -6.2 dB
# prefactor ratio). A model-derived, falsifiable prior -- not the raw d_min gap.
_PRICE_BRACKET_DB = (2.8, 4.0)


def _nn_predicted_price(target: float) -> float:
    """NN-approximation-predicted Es/N0 price (dB) of C_rd vs filled hex-42 at a
    target SER: the SAME model sim03 uses for the hex-vs-square gains, applied to
    this label-free pair on the Es/N0 axis (bits = 1). Positive = C_rd needs
    more Es/N0. This is the quantitative basis for the pre-registered bracket."""
    def _q(x: float) -> float:
        return 0.5 * math.erfc(x / math.sqrt(2.0))

    def _e(d2: float, kbar: float) -> float:
        lo, hi = -10.0, 90.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            s2 = t._noise_sigma_sq(mid, 1.0)          # bits = 1 -> Es/N0 axis
            if kbar * _q(math.sqrt(d2) / math.sqrt(2.0 * s2)) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    rd = _EXPECTED["radial-dual-r12-Nle60"]
    hx = _EXPECTED["hex-any-42"]
    return _e(float(rd["d2"]), rd["nn"] / 42) - _e(float(hx["d2"]), hx["nn"] / 42)


# ---------------------------------------------------------------------------
# Paired channel sweep (identical discipline to simulation 02)
# ---------------------------------------------------------------------------

def _ser_with_ci(errors: int, trials: int, alpha: float
                 ) -> Tuple[float, float, float]:
    rate = errors / trials
    lo, hi = t.clopper_pearson(errors, trials, alpha)
    return rate, lo, hi


def run_sweep(rd: t.Constellation, hx: t.Constellation,
              esn0_list: List[float], trials: int, alpha: float,
              rng: np.random.Generator) -> Tuple[List[Dict[str, float]], bool]:
    """CRN-paired AWGN sweep of C_rd vs filled hex-42; returns (rows, ml_ok).

    Both decoders are asserted bitwise-identical to exhaustive ML at every
    point (the C1 tie-in); ``ml_ok`` is the AND over the whole sweep.
    """
    ctx_hx = t.make_hex_decode_context(hx)                    # O(1) fast path
    ctx_rd = t.make_folded_decode_context(rd, fold_inversion=True)  # folded

    rows: List[Dict[str, float]] = []
    ml_ok = True
    for esn0 in esn0_list:
        tx = rng.integers(0, rd.size, trials)     # shared index stream (CRN)
        rd_tx = rd.points_unit[tx]
        hx_tx = hx.points_unit[tx]

        # Identical complex-noise realization for both constellations
        # (common random numbers via RNG-state rewind). bits = 0 -> Es/N0.
        state = rng.bit_generator.state
        rx_rd = t.awgn(rd_tx, esn0, rd.bits_per_symbol, rng)
        rng.bit_generator.state = state
        rx_hx = t.awgn(hx_tx, esn0, hx.bits_per_symbol, rng)

        rd_idx, _fast_rd = t.decode_hex_folded(rx_rd, ctx_rd)
        hx_idx, _fast_hx = t.decode_hex_fast(rx_hx, ctx_hx)

        # C1 tie-in: both practical decoders equal exhaustive ML here.
        rd_ml = t.decode_ml(rx_rd, rd.points_unit)
        hx_ml = t.decode_ml(rx_hx, hx.points_unit)
        rd_matches = bool(np.all(rd_idx == rd_ml))
        hx_matches = bool(np.all(hx_idx == hx_ml))
        ml_ok = ml_ok and rd_matches and hx_matches

        b_rd = rd_idx != tx
        b_hx = hx_idx != tx
        rd_err = int(np.sum(b_rd))
        hx_err = int(np.sum(b_hx))
        n_rd_only = int(np.sum(b_rd & ~b_hx))     # C_rd wrong, hex-42 right
        n_hx_only = int(np.sum(~b_rd & b_hx))     # hex-42 wrong, C_rd right
        mcnemar_p = t.mcnemar_pvalue(n_rd_only, n_hx_only)

        rd_ser, rd_lo, rd_hi = _ser_with_ci(rd_err, trials, alpha)
        hx_ser, hx_lo, hx_hi = _ser_with_ci(hx_err, trials, alpha)

        if rd_hi < hx_lo:
            ser_sig = "rd<hex42"      # C_rd significantly BETTER (unexpected)
        elif rd_lo > hx_hi:
            ser_sig = "rd>hex42"      # C_rd significantly worse (expected)
        else:
            ser_sig = "tie"

        rows.append({
            "esn0_db": esn0,
            "rd_ser": rd_ser, "rd_ser_lo": rd_lo, "rd_ser_hi": rd_hi,
            "hex42_ser": hx_ser, "hex42_ser_lo": hx_lo, "hex42_ser_hi": hx_hi,
            "ser_sig": ser_sig,
            "rd_decoder_matches_ml": float(rd_matches),
            "hex42_decoder_matches_ml": float(hx_matches),
            "rd_sym_err": rd_err, "hex42_sym_err": hx_err,
            "n_rd_only_err": n_rd_only, "n_hex42_only_err": n_hx_only,
            "mcnemar_p": mcnemar_p, "trials": trials,
        })

    # Holm-Bonferroni across the Es/N0 grid on the exact McNemar p-values.
    reject, p_adj = t.holm_bonferroni([r["mcnemar_p"] for r in rows], alpha)
    for r, rej, pa in zip(rows, reject, p_adj):
        if rej and r["n_rd_only_err"] > r["n_hex42_only_err"]:
            r["ser_sig_holm"] = "rd>hex42"          # C_rd resolved worse
        elif rej and r["n_hex42_only_err"] > r["n_rd_only_err"]:
            r["ser_sig_holm"] = "rd<hex42"          # C_rd resolved better
        else:
            r["ser_sig_holm"] = "tie"
        r["mcnemar_p_holm"] = float(pa)
    return rows, ml_ok


def _interp_esn0_at_ser(rows: List[Dict[str, float]], key: str,
                        target: float) -> float | None:
    """Log-linear interpolation of the Es/N0 achieving a target SER (same rule
    as simulation 02); None if the target is not bracketed by the curve."""
    pts = [(r["esn0_db"], r[key]) for r in rows if r[key] > 0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            ly0, ly1, lt = math.log10(y0), math.log10(y1), math.log10(target)
            frac = (lt - ly0) / (ly1 - ly0)
            return x0 + frac * (x1 - x0)
    return None


def price_with_ci(rows: List[Dict[str, float]], target: float
                  ) -> Dict[str, object]:
    """Es/N0 price of C_rd vs hex-42 at a target SER (positive = C_rd needs
    more Es/N0, i.e. the geometry price), with a Clopper-Pearson-derived band.

    Conservative (lower) price pits C_rd at its best curve (rd_ser_lo) against
    hex-42 at its worst (hex42_ser_hi); the optimistic (upper) price reverses
    that. ``ci_resolved`` is True iff even the conservative price is positive.
    """
    e_rd = _interp_esn0_at_ser(rows, "rd_ser", target)
    e_hx = _interp_esn0_at_ser(rows, "hex42_ser", target)
    out: Dict[str, object] = {
        "esn0_rd_db": e_rd, "esn0_hex42_db": e_hx, "price_db": None,
        "price_lo_db": None, "price_hi_db": None,
        "bracketed": e_rd is not None and e_hx is not None,
        "ci_resolved": False,
    }
    if not out["bracketed"]:
        return out
    out["price_db"] = e_rd - e_hx
    e_rd_best = _interp_esn0_at_ser(rows, "rd_ser_lo", target)
    e_rd_worst = _interp_esn0_at_ser(rows, "rd_ser_hi", target)
    e_hx_best = _interp_esn0_at_ser(rows, "hex42_ser_lo", target)
    e_hx_worst = _interp_esn0_at_ser(rows, "hex42_ser_hi", target)
    if e_rd_best is not None and e_hx_worst is not None:
        out["price_lo_db"] = e_rd_best - e_hx_worst
        out["ci_resolved"] = out["price_lo_db"] > 0
    if e_rd_worst is not None and e_hx_best is not None:
        out["price_hi_db"] = e_rd_worst - e_hx_best
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r_sq", type=int, default=12,
                    help="inversion radius^2 of the radial-dual constellation")
    ap.add_argument("--max_norm_sq", type=int, default=60,
                    help="largest shell norm in the radial-dual constellation")
    ap.add_argument("--esn0", type=float, nargs="+",
                    default=[6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
                             30, 32],
                    help="Es/N0 grid in dB (both constellations are label-free, "
                         "so the SNR axis is Es/N0; must bracket the target SERs "
                         "for BOTH curves)")
    ap.add_argument("--trials", type=int, default=100_000)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="1-alpha Clopper-Pearson confidence level")
    ap.add_argument("--targets", type=float, nargs="+", default=[1e-2, 1e-3],
                    help="target SER values for the geometry-price readout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 84)
    print("SIMULATION 08 -- AWGN geometry PRICE of the radial-dual "
          "constellation (C7 honesty)")
    print(f"seed={args.seed}  trials={args.trials}  M=42 vs M=42 at matched "
          f"average energy  CI={100 * (1 - args.alpha):.0f}%")
    print("=" * 84)

    rd = t.build_radial_dual_constellation(args.r_sq, args.max_norm_sq)
    hx = t.build_filled_constellation_any(rd.size)
    assert rd.size == hx.size == 42, "matched-order comparison requires M=42"

    prov_extra = {
        "candidate": {"name": rd.name, "M": rd.size,
                      "shell_norms": list(rd.shell_norms),
                      "inversion_r_sq": rd.inversion_r_sq,
                      "scale_sq_exact": str(rd.scale_sq_exact)},
        "baseline": {"name": hx.name, "M": hx.size,
                     "scale_sq_exact": str(hx.scale_sq_exact)},
        "snr_axis": "esn0_db (both constellations are label-free; "
                    "Eb/N0 = Es/N0 - 10*log10(log2 42) ~= Es/N0 - 7.29 dB "
                    "if a log2(42)-bit labeling were attached)",
        "price_bracket_db_at_1e-2": list(_PRICE_BRACKET_DB),
    }
    t.emit_provenance(args.results_dir, "sim08", args=args, extra=prov_extra)

    # ---- exact geometry preamble (pre-registered; == checks) ----------------
    print("\n[exact geometry] both objects at unit average energy "
          "(exact rationals, checked with ==):")
    print(f"   {'constellation':<24}{'d_min^2':>10}{'K_bar':>8}{'PAPR':>8}"
          f"{'PAPR(dB)':>10}{'==':>4}")
    geom_ok = True
    facts = {}
    for con in (rd, hx):
        d2, nn, papr = _exact_geometry(con)
        exp = _EXPECTED[con.name]
        ok = (d2 == exp["d2"] and nn == exp["nn"] and papr == exp["papr"])
        geom_ok = geom_ok and ok
        facts[con.name] = d2
        print(f"   {con.name:<24}{str(d2):>10}{nn / con.size:>8.3f}"
              f"{str(papr):>8}{10 * math.log10(float(papr)):>10.2f}"
              f"{('ok' if ok else 'FAIL'):>4}")
    pure_penalty = 10.0 * math.log10(float(facts[hx.name]) /
                                     float(facts[rd.name]))
    print(f"   pure-d_min^2 penalty of C_rd vs filled hex-42: "
          f"{pure_penalty:+.2f} dB (= 10*log10(128/41), exact inputs)")
    print(f"   NN-approximation predicted price @ SER 1e-2: "
          f"{_nn_predicted_price(1e-2):+.2f} dB (the +4.94 dB d_min penalty "
          f"folded with C_rd's much lower K_bar, 1.14 vs 4.76)")
    print(f"   pre-registered bracket for the MEASURED Es/N0 gap @ SER 1e-2: "
          f"{_PRICE_BRACKET_DB[0]:.1f}-{_PRICE_BRACKET_DB[1]:.1f} dB "
          f"(centred on the model prediction above; falsifiable).")

    # ---- paired sweep --------------------------------------------------------
    rows, ml_ok = run_sweep(rd, hx, list(args.esn0), args.trials,
                            args.alpha, rng)
    print(f"\n[C1 tie-in] folded decoder (C_rd) == ML and fast decoder "
          f"(hex-42) == ML at every point: {'PASS' if ml_ok else 'FAIL'}")

    print(f"\n{'Es/N0':>6} {'C_rd SER':>10} {'hex42 SER':>10} {'cmp':>9} "
          f"{'cmp(Holm)':>10} {'McNemar p':>10}")
    for r in rows:
        print(f"{r['esn0_db']:>6.1f} {r['rd_ser']:>10.3e} "
              f"{r['hex42_ser']:>10.3e} {r['ser_sig']:>9} "
              f"{r['ser_sig_holm']:>10} {r['mcnemar_p']:>10.2e}")
    n_worse = sum(1 for r in rows if r["ser_sig_holm"] == "rd>hex42")
    n_better = sum(1 for r in rows if r["ser_sig_holm"] == "rd<hex42")
    n_tie = len(rows) - n_worse - n_better
    print(f"   [paired McNemar + Holm @ alpha={args.alpha:g}: C_rd worse at "
          f"{n_worse}/{len(rows)} points, better at {n_better}, tie at "
          f"{n_tie}. 'Worse' is the EXPECTED, pre-registered direction: this "
          f"is the geometry price, stated up front.]")

    # ---- price at target -----------------------------------------------------
    summary: List[Dict[str, object]] = []
    print("\n[geometry price] Es/N0 gap at target SER "
          "(positive = C_rd needs more Es/N0):")
    for target in args.targets:
        g = price_with_ci(rows, target)
        rec: Dict[str, object] = {"target_ser": target}
        rec.update(g)
        rec.update({
            "nn_pred_price_db": _nn_predicted_price(target),
            "rd_d2min_exact": str(_EXPECTED[rd.name]["d2"]),
            "hex42_d2min_exact": str(_EXPECTED[hx.name]["d2"]),
            "pure_dmin_penalty_db": pure_penalty,
            "rd_papr_db": 10 * math.log10(float(_EXPECTED[rd.name]["papr"])),
            "hex42_papr_db": 10 * math.log10(float(_EXPECTED[hx.name]["papr"])),
        })
        summary.append(rec)
        if not g["bracketed"]:
            print(f"   @ SER={target:.0e}: not bracketed by the grid -- "
                  f"extend --esn0 or raise --trials")
            continue
        band = ""
        if g["price_lo_db"] is not None and g["price_hi_db"] is not None:
            band = (f"  [CI {g['price_lo_db']:+.2f}..{g['price_hi_db']:+.2f} dB,"
                    f" resolved={g['ci_resolved']}]")
        print(f"   @ SER={target:.0e}: price {g['price_db']:+.2f} dB "
              f"(C_rd {g['esn0_rd_db']:.2f} vs hex-42 "
              f"{g['esn0_hex42_db']:.2f} dB Es/N0){band}")
        if target == 1e-2:
            lo, hi = _PRICE_BRACKET_DB
            inside = lo <= g["price_db"] <= hi
            print(f"   pre-registration verdict @ 1e-2: measured "
                  f"{g['price_db']:+.2f} dB is "
                  f"{'INSIDE' if inside else 'OUTSIDE'} the expected "
                  f"{lo:.1f}-{hi:.1f} dB bracket "
                  f"({'as pre-registered' if inside else 'reported honestly'}); "
                  f"NN-approximation predicted {_nn_predicted_price(1e-2):+.2f} dB.")

    print("\n  C7-HONESTY RESULT: "
          f"{'PASS' if (geom_ok and ml_ok) else 'FAIL'} "
          f"(exact geometry ==; both decoders bitwise-ML; the price is "
          f"measured and stated, not hidden)")
    print("  [Scope: this NULL-TO-NEGATIVE geometry result qualifies C7. C_rd's")
    print("   value is the exactness/structure column -- the 6x / 10.5x storage")
    print("   folds and the order-12 codec -- and the price of that structure in")
    print("   AWGN error-rate geometry is the number above. No claim column is")
    print("   traded against another.]")

    # ---- CSVs ----------------------------------------------------------------
    ser_path = os.path.join(args.results_dir, "sim08_ser.csv")
    with open(ser_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sum_cols = ["target_ser", "esn0_rd_db", "esn0_hex42_db", "price_db",
                "price_lo_db", "price_hi_db", "ci_resolved", "bracketed",
                "nn_pred_price_db", "rd_d2min_exact", "hex42_d2min_exact",
                "pure_dmin_penalty_db", "rd_papr_db", "hex42_papr_db"]
    sum_path = os.path.join(args.results_dir, "sim08_gap_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {ser_path}")
    print(f"Wrote {sum_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
