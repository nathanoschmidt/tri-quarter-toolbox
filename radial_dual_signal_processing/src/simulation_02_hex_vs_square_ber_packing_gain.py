#!/usr/bin/env python3
"""
simulation_02_hex_vs_square_ber_packing_gain.py - Hexagonal vs. Square-QAM BER/SER Packing Gain (C3)

Compares the hexagonal constellation against square QAM, at matched order and
energy, for the Tri-Quarter Framework (TQF) radial_dual_signal_processing
subproject.

Claim validated
---------------
C3 (Packing gain): at matched constellation order M and matched average symbol
    energy, the hexagonal constellation attains a lower symbol-error rate (SER)
    than square QAM under AWGN, approaching the ~0.6 dB asymptotic packing gain
    (the gain grows with M). Under flat Rayleigh fading (perfect CSI) the gain
    persists but is smaller; under heavy 2D impulsive noise the impulse error
    floor masks the geometric gain, so that channel demonstrates hex/square
    robustness PARITY rather than a packing gain. SER (labeling-independent) is
    the headline metric; BER is secondary because the hex labeling is only
    Gray-LIKE (so hex BER can exceed square's true-Gray BER).

Methodology
-----------
* Apples-to-apples: same M, unit average energy for both constellations, the
  same Eb/N0 grid, and common random numbers -- the identical data stream and
  the identical noise/fading realizations drive both constellations (achieved
  by rewinding the RNG state between the two channel calls).
* SER is the headline metric (it depends only on geometry and the ML-equivalent
  decoder, so it is labeling-independent). BER is also reported: square QAM uses
  a true Gray map; the hexagonal constellation uses a documented spatial
  Gray-like labeling, so its BER is secondary and labeling-dependent.
* The hexagonal constellation is decoded with the closed-form O(1) decoder,
  which is verified bitwise-identical to exhaustive ML on the stream (C1 tie-in).
* Exact Clopper-Pearson binomial confidence intervals are reported on every
  SER/BER point.
* Per-point significance: because the stream is common-random-number paired, the
  hex-vs-square difference at each Eb/N0 is tested with an exact McNemar test on
  the discordant error pairs, Holm-corrected across the grid (the resolved flag
  ``ser_sig_holm`` and ``sim02_significance_summary.csv``). The marginal
  Clopper-Pearson CI-overlap flag (``ser_sig``) is retained as a conservative
  companion. This paired test is the resolved read under flat Rayleigh fading,
  where the dB-at-target gain is ill-conditioned (the flat 1/SNR roll-off
  inflates the SNR-at-target variance) while hex is still point-wise better.

Output: the per-channel tables and the packing-gain summary.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.2.0
Date: July 4, 2026

Reproducibility invariant: this script's RNG consumption order is load-bearing.
The committed sim02_*.csv files reproduce byte-identically under seed 42 only if
no computation, loop order, or random draw is changed; treat the draw sequence
as fixed when editing.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Callable, Dict, List, Tuple

import numpy as np

import tqf_hex_signal as t


def _ser_with_ci(errors: int, trials: int, alpha: float) -> Tuple[float, float, float]:
    rate = errors / trials
    lo, hi = t.clopper_pearson(errors, trials, alpha)
    return rate, lo, hi


def _decode_received(received: np.ndarray, gains: np.ndarray | None,
                     con: t.Constellation, ctx, is_hex: bool) -> np.ndarray:
    """Decode received symbols (zero-forcing first if fading gains are given)."""
    z = received if gains is None else received / gains
    if is_hex:
        idx, _fast = t.decode_hex_fast(z, ctx)
        return idx
    return t.decode_ml(z, con.points_unit)


def run_channel(channel: str, m: int, ebn0_list: List[float], trials: int,
                alpha: float, p: float, amplitude: float,
                rng: np.random.Generator) -> List[Dict[str, float]]:
    """Run the hex-vs-square comparison for one channel and one M over the grid."""
    hex_con = t.build_filled_constellation(m)
    sq_con = t.build_square_qam(m)
    hex_ctx = t.make_hex_decode_context(hex_con)
    nbits = hex_con.bits_per_symbol

    rows: List[Dict[str, float]] = []
    for ebn0 in ebn0_list:
        tx = rng.integers(0, m, trials)               # shared data stream (CRN)
        hex_tx = hex_con.points_unit[tx]
        sq_tx = sq_con.points_unit[tx]

        # Apply the identical channel realization to both constellations by
        # rewinding the RNG state between the two calls (common random numbers).
        state = rng.bit_generator.state
        if channel == "awgn":
            rx_hex = t.awgn(hex_tx, ebn0, nbits, rng); h_hex = None
            rng.bit_generator.state = state
            rx_sq = t.awgn(sq_tx, ebn0, nbits, rng); h_sq = None
        elif channel == "impulsive":
            rx_hex = t.impulsive(hex_tx, ebn0, nbits, rng, p, amplitude); h_hex = None
            rng.bit_generator.state = state
            rx_sq = t.impulsive(sq_tx, ebn0, nbits, rng, p, amplitude); h_sq = None
        elif channel == "rayleigh":
            rx_hex, h_hex = t.rayleigh(hex_tx, ebn0, nbits, rng)
            rng.bit_generator.state = state
            rx_sq, h_sq = t.rayleigh(sq_tx, ebn0, nbits, rng)
        else:
            raise ValueError(f"unknown channel {channel!r}")

        hex_idx = _decode_received(rx_hex, h_hex, hex_con, hex_ctx, is_hex=True)
        sq_idx = _decode_received(rx_sq, h_sq, sq_con, None, is_hex=False)

        # C1 tie-in: confirm the O(1) hex decoder equals exhaustive ML here.
        z_hex = rx_hex if h_hex is None else rx_hex / h_hex
        hex_ml = t.decode_ml(z_hex, hex_con.points_unit)
        hex_decoder_matches_ml = bool(np.all(hex_idx == hex_ml))

        # Paired (common-random-number) error indicators: because hex and square
        # saw the identical noise/fading realization, the per-symbol agreement is
        # meaningful and the discordant pairs drive an exact McNemar test below.
        b_hex = hex_idx != tx
        b_sq = sq_idx != tx
        hex_sym_err = int(np.sum(b_hex))
        sq_sym_err = int(np.sum(b_sq))
        n_hex_only_err = int(np.sum(b_hex & ~b_sq))   # hex wrong, square right
        n_sq_only_err = int(np.sum(~b_hex & b_sq))    # square wrong, hex right
        mcnemar_p = t.mcnemar_pvalue(n_sq_only_err, n_hex_only_err)
        hex_bit_err = t.hamming_bits(hex_con.labels[tx], hex_con.labels[hex_idx], nbits)
        sq_bit_err = t.hamming_bits(sq_con.labels[tx], sq_con.labels[sq_idx], nbits)

        hex_ser, hex_ser_lo, hex_ser_hi = _ser_with_ci(hex_sym_err, trials, alpha)
        sq_ser, sq_ser_lo, sq_ser_hi = _ser_with_ci(sq_sym_err, trials, alpha)
        nbit_total = trials * nbits
        hex_ber, hex_ber_lo, hex_ber_hi = _ser_with_ci(hex_bit_err, nbit_total, alpha)
        sq_ber, sq_ber_lo, sq_ber_hi = _ser_with_ci(sq_bit_err, nbit_total, alpha)

        # Is the SER difference resolved by the (exact) Clopper-Pearson CIs?
        if hex_ser_hi < sq_ser_lo:
            ser_sig = "hex<sq"     # hex significantly better (CIs disjoint)
        elif hex_ser_lo > sq_ser_hi:
            ser_sig = "hex>sq"     # hex significantly worse (CIs disjoint)
        else:
            ser_sig = "tie"        # CIs overlap: difference not statistically resolved

        rows.append({
            "ebn0_db": ebn0,
            "hex_ser": hex_ser, "hex_ser_lo": hex_ser_lo, "hex_ser_hi": hex_ser_hi,
            "sq_ser": sq_ser, "sq_ser_lo": sq_ser_lo, "sq_ser_hi": sq_ser_hi,
            "ser_sig": ser_sig,
            "hex_ber": hex_ber, "hex_ber_lo": hex_ber_lo, "hex_ber_hi": hex_ber_hi,
            "sq_ber": sq_ber, "sq_ber_lo": sq_ber_lo, "sq_ber_hi": sq_ber_hi,
            "hex_decoder_matches_ml": float(hex_decoder_matches_ml),
            "hex_sym_err": hex_sym_err, "sq_sym_err": sq_sym_err,
            "n_hex_only_err": n_hex_only_err, "n_sq_only_err": n_sq_only_err,
            "mcnemar_p": mcnemar_p, "trials": trials,
        })

    # Paired per-point significance: Holm-Bonferroni across the Eb/N0 grid on the
    # exact McNemar p-values. The stream is common-random-number paired, so the
    # discordant-pair (McNemar) test is the correct, more powerful per-point
    # comparison; the marginal CI-overlap flag (ser_sig) is kept as a companion.
    reject, p_adj = t.holm_bonferroni([r["mcnemar_p"] for r in rows], alpha)
    for r, rej, pa in zip(rows, reject, p_adj):
        if rej and r["n_sq_only_err"] > r["n_hex_only_err"]:
            r["ser_sig_holm"] = "hex<sq"          # hex resolved better (Holm)
        elif rej and r["n_hex_only_err"] > r["n_sq_only_err"]:
            r["ser_sig_holm"] = "hex>sq"          # hex resolved worse (Holm)
        else:
            r["ser_sig_holm"] = "tie"             # not resolved after correction
        r["mcnemar_p_holm"] = float(pa)
    return rows


def _interp_ebn0_at_ser(rows: List[Dict[str, float]], key_ser: str,
                        target: float) -> float | None:
    """Linear interpolation (Eb/N0 vs log10 SER) of the Eb/N0 achieving a target
    SER; returns None if the target is not bracketed by the measured curve."""
    pts = [(r["ebn0_db"], r[key_ser]) for r in rows if r[key_ser] > 0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - target) * (y1 - target) <= 0 and y0 != y1:
            ly0, ly1, lt = math.log10(y0), math.log10(y1), math.log10(target)
            frac = (lt - ly0) / (ly1 - ly0)
            return x0 + frac * (x1 - x0)
    return None


def _gain_with_ci(rows: List[Dict[str, float]], target: float) -> Dict[str, object]:
    """dB packing gain (eb_sq - eb_hex; positive => hex better) at a target SER,
    with a confidence band derived from the per-point Clopper-Pearson SER bounds.

    The band brackets the gain by interpolating the optimistic/pessimistic SER
    curves: the conservative (lower) gain pits square at its best (sq_ser_lo)
    against hex at its worst (hex_ser_hi); the optimistic (upper) gain does the
    reverse. ``ci_resolved`` is True iff the conservative gain is still positive,
    i.e. the advantage survives the CIs.
    """
    eb_hex = _interp_ebn0_at_ser(rows, "hex_ser", target)
    eb_sq = _interp_ebn0_at_ser(rows, "sq_ser", target)
    out: Dict[str, object] = {
        "eb_hex_db": eb_hex, "eb_sq_db": eb_sq, "gain_db": None,
        "gain_lo_db": None, "gain_hi_db": None,
        "bracketed": eb_hex is not None and eb_sq is not None, "ci_resolved": False,
    }
    if not out["bracketed"]:
        return out
    out["gain_db"] = eb_sq - eb_hex
    eb_hex_worst = _interp_ebn0_at_ser(rows, "hex_ser_hi", target)
    eb_hex_best = _interp_ebn0_at_ser(rows, "hex_ser_lo", target)
    eb_sq_best = _interp_ebn0_at_ser(rows, "sq_ser_lo", target)
    eb_sq_worst = _interp_ebn0_at_ser(rows, "sq_ser_hi", target)
    if eb_sq_best is not None and eb_hex_worst is not None:
        out["gain_lo_db"] = eb_sq_best - eb_hex_worst
        out["ci_resolved"] = out["gain_lo_db"] > 0
    if eb_sq_worst is not None and eb_hex_best is not None:
        out["gain_hi_db"] = eb_sq_worst - eb_hex_best
    return out


def _impulsive_floor(rows: List[Dict[str, float]], m: int, p: float,
                     tail: int = 3) -> Dict[str, object]:
    """Measured high-SNR SER floor (mean over the last ``tail`` points) and the
    impulse rate it implies via floor = p*(1-1/M), alongside the theoretical floor."""
    hex_floor = float(np.mean([r["hex_ser"] for r in rows[-tail:]]))
    sq_floor = float(np.mean([r["sq_ser"] for r in rows[-tail:]]))
    denom = (1.0 - 1.0 / m)
    return {
        "measured_floor_hex": hex_floor,
        "measured_floor_sq": sq_floor,
        "implied_p": (hex_floor / denom) if denom else float("nan"),
        "theoretical_floor": p * denom,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--channels", type=str, nargs="+",
                    default=["awgn", "impulsive", "rayleigh"],
                    choices=["awgn", "impulsive", "rayleigh"])
    ap.add_argument("--ebn0", type=float, nargs="+",
                    default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                    help="Eb/N0 grid in dB (extend higher, e.g. up to 30, for "
                         "low-SER targets under Rayleigh fading)")
    ap.add_argument("--rayleigh_ebn0_extra", type=float, nargs="*",
                    default=[26, 28, 30, 32, 34, 36, 38, 40, 42, 44],
                    help="extra high-Eb/N0 points appended for the rayleigh channel "
                         "only (its slow 1/SNR roll-off needs ~40 dB to reach SER "
                         "1e-3 at M>=64). Appended after the base grid; pass empty "
                         "to disable. Rayleigh must be the last channel processed so "
                         "this does not perturb the AWGN/impulsive RNG stream.")
    ap.add_argument("--trials", type=int, default=100_000)
    ap.add_argument("--p", type=float, default=0.1, help="impulsive hit probability")
    ap.add_argument("--amplitude", type=float, default=5.0, help="impulse magnitude")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="1-alpha Clopper-Pearson confidence level")
    ap.add_argument("--targets", type=float, nargs="+", default=[1e-2, 1e-3],
                    help="target SER values for the packing-gain readout")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 78)
    print("SIMULATION 02 -- hexagonal vs square QAM: SER/BER and packing gain (C3)")
    print(f"seed={args.seed}  M={args.M}  trials={args.trials}  channels={args.channels}")
    print(f"impulsive p={args.p} A={args.amplitude}  CI={100*(1-args.alpha):.0f}%")
    print("=" * 78)
    t.emit_provenance(args.results_dir, "sim02", args=args)

    # Determinism guard: the rayleigh grid extension consumes extra RNG, so
    # rayleigh must run LAST or it would perturb the AWGN/impulsive streams that
    # the headline results depend on. Reorder defensively without dropping any
    # channel the user requested.
    channels = sorted(args.channels, key=lambda c: c == "rayleigh")

    all_match_ml = True
    summary: List[Dict[str, object]] = []
    sig_summary: List[Dict[str, object]] = []
    for channel in channels:
        ebn0_grid = list(args.ebn0)
        if channel == "rayleigh" and args.rayleigh_ebn0_extra:
            ebn0_grid = list(args.ebn0) + list(args.rayleigh_ebn0_extra)
        for m in args.M:
            rows = run_channel(channel, m, ebn0_grid, args.trials,
                               args.alpha, args.p, args.amplitude, rng)
            match_ml = all(r["hex_decoder_matches_ml"] > 0.5 for r in rows)
            all_match_ml = all_match_ml and match_ml

            print(f"\n--- channel={channel}  M={m}  "
                  f"(hex O(1) decoder == ML: {match_ml}) ---")
            print(f"{'Eb/N0':>6} {'hexSER':>10} {'sqSER':>10} {'SERcmp':>8} "
                  f"{'hexBER':>10} {'sqBER':>10}")
            for r in rows:
                print(f"{r['ebn0_db']:>6.1f} {r['hex_ser']:>10.3e} {r['sq_ser']:>10.3e} "
                      f"{r['ser_sig']:>8} {r['hex_ber']:>10.3e} {r['sq_ber']:>10.3e}")
            print("   [SER is the C3 metric (geometry, labeling-independent). "
                  "SERcmp flags whether the")
            print("    hex/square SER CIs are disjoint ('hex<sq' = hex better) or "
                  "overlap ('tie').")
            print("    BER is secondary: square uses a true Gray map; hex uses a "
                  "heuristic Gray-like labeling.]")

            # Paired per-point significance roll-up (McNemar + Holm). This is the
            # resolved read for channels where the dB-at-target gain is ill-
            # conditioned (e.g. Rayleigh's flat 1/SNR roll-off): it counts the
            # Eb/N0 points at which hex is significantly better after correction.
            n_better = sum(1 for r in rows if r["ser_sig_holm"] == "hex<sq")
            n_worse = sum(1 for r in rows if r["ser_sig_holm"] == "hex>sq")
            n_tie = len(rows) - n_better - n_worse
            min_p = min(r["mcnemar_p"] for r in rows)
            print(f"   [paired McNemar + Holm @ alpha={args.alpha:g}: hex<sq at "
                  f"{n_better}/{len(rows)} Eb/N0 points, hex>sq at {n_worse}, "
                  f"tie at {n_tie}; min McNemar p={min_p:.2e}]")
            sig_summary.append({
                "channel": channel, "M": m, "n_points": len(rows),
                "n_hex_better_holm": n_better, "n_hex_worse_holm": n_worse,
                "n_tie_holm": n_tie, "min_mcnemar_p": min_p,
                "alpha": args.alpha, "method": "mcnemar+holm", "crn": True,
            })

            # Packing-gain readout at each target SER (point estimate + CI band).
            floor_info = (_impulsive_floor(rows, m, args.p)
                          if channel == "impulsive" else {})
            for target in args.targets:
                g = _gain_with_ci(rows, target)
                rec = {"channel": channel, "M": m, "target_ser": target}
                rec.update(g)
                rec.update(floor_info)
                summary.append(rec)
                if not g["bracketed"]:
                    why = ("error floor above target -> robustness parity"
                           if channel == "impulsive"
                           else "extend --ebn0 or raise --trials")
                    print(f"   packing gain @ SER={target:.0e}: not bracketed ({why})")
                else:
                    band = ""
                    if g["gain_lo_db"] is not None and g["gain_hi_db"] is not None:
                        band = (f"  [CI {g['gain_lo_db']:+.2f}..{g['gain_hi_db']:+.2f} dB, "
                                f"resolved={g['ci_resolved']}]")
                    print(f"   packing gain @ SER={target:.0e}: {g['gain_db']:+.2f} dB "
                          f"(hex {g['eb_hex_db']:.2f} vs square {g['eb_sq_db']:.2f} dB){band}")

            if channel == "impulsive":
                print(f"   NOTE: impulsive SER floor ~ p*(1-1/M) = "
                      f"{floor_info['theoretical_floor']:.3f}; measured "
                      f"{floor_info['measured_floor_hex']:.3f} (implied p="
                      f"{floor_info['implied_p']:.3f}). The geometric gain cannot")
                print(f"   lower this floor, so at the floor hex/square show ROBUSTNESS "
                      f"PARITY (expect 'tie'); any hex<sq is the pre-floor waterfall.")

            csv_path = os.path.join(args.results_dir, f"sim02_ber_{channel}_M{m}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # Durable C3 headline artifact: the dB packing gain (with CI band) per
    # (channel, M, target), plus the impulsive floor / implied p.
    sum_cols = ["channel", "M", "target_ser", "eb_hex_db", "eb_sq_db", "gain_db",
                "gain_lo_db", "gain_hi_db", "ci_resolved", "bracketed",
                "measured_floor_hex", "measured_floor_sq", "implied_p",
                "theoretical_floor"]
    sum_path = os.path.join(args.results_dir, "sim02_gain_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    # Durable C3 companion artifact: the paired per-point significance roll-up
    # (McNemar + Holm). This carries the resolved Rayleigh read, where the dB-at-
    # target gain CI is ill-conditioned but the paired test still resolves hex.
    sig_cols = ["channel", "M", "n_points", "n_hex_better_holm",
                "n_hex_worse_holm", "n_tie_holm", "min_mcnemar_p",
                "alpha", "method", "crn"]
    sig_path = os.path.join(args.results_dir, "sim02_significance_summary.csv")
    with open(sig_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sig_cols)
        w.writeheader()
        w.writerows(sig_summary)

    print(f"\nAll hex O(1) decoders matched exhaustive ML: {all_match_ml}")
    print(f"Wrote per-(channel, M) CSVs to {args.results_dir}/sim02_ber_*.csv")
    print(f"Wrote {sum_path}  (dB gain + CI band + impulsive floor)")
    print(f"Wrote {sig_path}  (paired McNemar + Holm per-point significance)")
    print("=" * 78)


if __name__ == "__main__":
    main()
