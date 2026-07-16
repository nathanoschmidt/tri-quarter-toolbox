#!/usr/bin/env python3
"""
simulation_04_phase_rotation_and_differential.py - Study 4 (Episode II).

Phase-rotation robustness and differential hexagonal decode (claims C6, C8).

Demonstrates decoder equivariance and differential-hexagonal phase robustness for
the Tri-Quarter Framework (TQF) radial_dual_signal_processing subproject.

Claim validated
---------------
C6 (Equivariant / rotation-robust decode): rotation by pi/3 is a symmetry of the
    hexagonal constellation -- a phase rotation by k*pi/3 permutes the sector
    index by +k (mod 6). A differential hexagonal scheme that carries data in the
    difference of consecutive sector indices (mod 6) is therefore immune to any
    static carrier-phase offset that is a multiple of pi/3 (the hexagonal
    analogue of DPSK), whereas coherent decoding suffers a catastrophic
    sector-slip as the offset approaches 60 degrees.

Demonstration vehicle: a six-point constellation with one unit-energy symbol at
the center of each angular sector (the cleanest carrier of the sector-rotation
story). Coherent and differential schemes are compared over a residual
phase-offset sweep under AWGN, with common random numbers (identical data and
noise for both schemes at each offset). The differential scheme returns to the
noise floor at offsets of 0 AND 60 degrees; the coherent scheme degrades to
near-certain error around 60 degrees. The cost is the differential detection
penalty at zero offset, which the script reports.

Scope: the six-point constellation is a DIDACTIC instance (differential 6-PSK)
chosen because a pi/3 rotation maps each symbol exactly onto the next. The
underlying equivariance -- the closed-form decoder commutes with the order-6
rotation, permuting the sector index by +1 -- is a property of the 2D hex
lattice itself; the script verifies it EXACTLY on a real 6-fold-symmetric hex
(disk) constellation before the channel sweep. The sector-index differential
idea generalizes to any hex constellation via the senary label s_6, with the
caveat that a pi/3 rotation also permutes the within-sector (shell) structure,
which a complete scheme must additionally handle.

Output: the SER-vs-offset table and the zero-offset penalty line.

Eb/N0 convention (per information bit)
-------------------------------------
The senary constellation carries exactly log2(6) ~= 2.585 information bits per
symbol, so the --ebn0 flag is defined per information bit: the true float
log2(6) is passed through the noise mapping (for this constellation
Es/N0 = Eb/N0 + 10*log10(log2 6) ~= Eb/N0 + 4.12 dB). The exact zero-noise
equivariance and differential round-trip checks below use no channel and are
independent of this scaling.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Tuple

import numpy as np

import tqf_hex_signal as t


def _sector_center_constellation() -> np.ndarray:
    """Six unit-energy points, one at the center (30 + 60k degrees) of each sector."""
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.exp(1j * angles)


def _decode_sectors(received: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Nearest-point decode to a sector index in {0..5} (exhaustive over 6 points)."""
    return t.decode_ml(received, points)


def verify_equivariance_on_hex(max_norm_sq: int = 37) -> Tuple[int, int]:
    """Exact check that the DECODER commutes with the order-6 rotation on a real
    2D hexagonal constellation (not just the 6-PSK demo).

    Uses the 6-fold-symmetric disk constellation (closed under R). Rotates every
    constellation point by +pi/3 in signal space, decodes the rotated point with
    the closed-form hex decoder, and verifies the decoded point's integer sector
    index equals (original sector + 1) mod 6. Returns (num_points, violations);
    violations == 0 confirms the equivariance theorem underlying C6 on the
    actual lattice constellation.
    """
    con = t.build_disk_constellation(max_norm_sq)
    ctx = t.make_hex_decode_context(con)
    ab = con.ab
    sec0 = t.sector_index_array(ab)                       # exact integer sectors
    rotated = con.points_unit * np.exp(1j * math.pi / 3.0)
    idx, _fast = t.decode_hex_fast(rotated, ctx)          # decode rotated points
    sec1 = t.sector_index_array(ab[idx])                  # sectors of decoded points
    violations = int(np.sum(sec1 != (sec0 + 1) % 6))
    return ab.shape[0], violations


def verify_differential_roundtrip(trials: int = 2000) -> bool:
    """At zero noise, differential encode/decode recovers the data EXACTLY for
    every static offset that is a multiple of pi/3 (the hex-DPSK invariance)."""
    points = _sector_center_constellation()
    rng = np.random.default_rng(0)
    data = rng.integers(0, 6, trials)
    diff_sectors = np.empty(trials + 1, dtype=np.int64)
    diff_sectors[0] = 0
    diff_sectors[1:] = np.cumsum(data) % 6
    ok = True
    for kdeg in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0):
        rx = points[diff_sectors] * np.exp(1j * math.radians(kdeg))  # zero noise
        dec = t.decode_ml(rx, points)
        data_hat = (dec[1:] - dec[:-1]) % 6
        ok = ok and bool(np.array_equal(data_hat, data))
    return ok


def verify_t24_differential_invariance(trials: int, seed: int
                                       ) -> Tuple[int, List[dict]]:
    """Exact invariance of the combined rotation + inversion (Z6 x Z2) differential
    codec under all 12 static (rotation k*pi/3, inversion m) actions.

    The T24 codec carries data as a (sector in Z6, inversion-bit in Z2) pair and
    transmits consecutive differences of each component. Any static action -- a
    constant sector offset k and/or a global inversion flip m, applied uniformly
    to the whole stream -- cancels in the receiver's differences. This is a
    label-domain construction (the inversion bit is a discrete state, never a
    Euclidean operation), so the check is exact: every one of the 12 actions must
    recover the data with zero error from the second symbol onward.

    Returns (total_violations, per_action_rows).
    """
    rng = np.random.default_rng(seed)
    d_sec = rng.integers(0, 6, trials)
    d_inv = rng.integers(0, 2, trials)
    s, u = t.differential_encode_t24(d_sec, d_inv)
    rows: List[dict] = []
    total_violations = 0
    for k in range(6):                      # static rotation by k*60 deg
        for m in (0, 1):                    # global amplitude inversion or not
            s2 = (s + k) % 6                 # constant offset on the whole stream
            u2 = (u + m) % 2
            ds, du = t.differential_decode_t24(s2, u2)
            sec_ok = bool(np.array_equal(ds[1:], d_sec[1:]))
            inv_ok = bool(np.array_equal(du[1:], d_inv[1:]))
            viol = int(np.sum(ds[1:] != d_sec[1:]) + np.sum(du[1:] != d_inv[1:]))
            total_violations += viol
            rows.append({
                "rotation_k": k, "inversion_m": m,
                "sector_recovered": sec_ok, "inversion_recovered": inv_ok,
                "violations": viol,
            })
    return total_violations, rows


def run_offset_sweep(dtheta_deg: List[float], trials: int, ebn0_db: float,
                     seed: int) -> List[dict]:
    """Compare coherent vs differential-hex SER across a phase-offset sweep (CRN)."""
    points = _sector_center_constellation()
    k_bits = math.log2(6.0)          # exactly log2(6) information bits/symbol
    rng = np.random.default_rng(seed)

    # Common random data and noise, drawn once and reused at every offset and by
    # both schemes (the only thing that changes across rows is the phase offset).
    data = rng.integers(0, 6, trials)                 # senary information symbols
    # Differential transmitted sectors with a leading pilot (reference sector 0).
    diff_sectors = np.empty(trials + 1, dtype=np.int64)
    diff_sectors[0] = 0
    diff_sectors[1:] = (np.cumsum(data) ) % 6
    # Coherent transmits the data symbols directly (plus a matching pilot slot so
    # both schemes see identically-shaped streams and identical noise draws).
    coh_sectors = np.empty(trials + 1, dtype=np.int64)
    coh_sectors[0] = 0
    coh_sectors[1:] = data

    noise = (rng.standard_normal(trials + 1) + 1j * rng.standard_normal(trials + 1))
    # Pass the true fractional bits/symbol so Eb/N0 is per information bit
    # (rounding to an integer bit count would misstate the noise scale by ~0.65 dB).
    sigma = math.sqrt(t._noise_sigma_sq(ebn0_db, k_bits))
    noise = noise * (sigma / math.sqrt(2.0))          # CN(0, sigma^2)

    rows: List[dict] = []
    for deg in dtheta_deg:
        phase = np.exp(1j * math.radians(deg))

        # Coherent: decode absolute sector, no phase compensation.
        rx_coh = points[coh_sectors] * phase + noise
        dec_coh = _decode_sectors(rx_coh, points)
        coh_err = int(np.sum(dec_coh[1:] != data))    # exclude pilot slot

        # Differential: decode sectors, recover data as consecutive differences.
        rx_diff = points[diff_sectors] * phase + noise
        dec_diff = _decode_sectors(rx_diff, points)
        data_hat = (dec_diff[1:] - dec_diff[:-1]) % 6  # difference cancels offset
        diff_err = int(np.sum(data_hat != data))

        coh_ser = coh_err / trials
        diff_ser = diff_err / trials
        coh_lo, coh_hi = t.clopper_pearson(coh_err, trials)
        diff_lo, diff_hi = t.clopper_pearson(diff_err, trials)
        rows.append({
            "dtheta_deg": deg,
            "coherent_ser": coh_ser, "coherent_lo": coh_lo, "coherent_hi": coh_hi,
            "differential_ser": diff_ser, "differential_lo": diff_lo,
            "differential_hi": diff_hi,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtheta", type=float, nargs="+",
                    default=list(range(0, 71, 5)),
                    help="phase-offset sweep in degrees (should cross 60 deg)")
    ap.add_argument("--trials", type=int, default=100_000)
    ap.add_argument("--ebn0", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t24_trials", type=int, default=20_000,
                    help="trials for the C8 combined rotation+inversion (Z6 x Z2) "
                         "differential invariance check")
    ap.add_argument("--t24_seed", type=int, default=808,
                    help="independent seed for the C8 check (uses an "
                         "independent stream)")
    ap.add_argument("--results_dir", type=str, default=".")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 72)
    print("SIMULATION 04 -- phase-rotation robustness: coherent vs differential (C6)")
    es_n0 = args.ebn0 + 10.0 * math.log10(math.log2(6.0))
    print(f"seed={args.seed}  trials={args.trials}  Eb/N0={args.ebn0} dB per "
          f"information bit (log2(6) bits/symbol; Es/N0 = {es_n0:.2f} dB)  "
          f"6-point sector constellation")
    print("=" * 72)
    t.emit_provenance(args.results_dir, "sim04", args=args)

    # ---- Exact verifications on a REAL 2D hex constellation (before sweep) --
    npts, viol = verify_equivariance_on_hex()
    eq_pass = (viol == 0)
    rt_pass = verify_differential_roundtrip()
    print("\n[C6 verify] grounding the equivariance on the actual hex lattice "
          "(not only the 6-PSK demo):")
    print(f"   decoder equivariance: rotating all {npts} points of a 6-fold-"
          f"symmetric hex (disk)")
    print(f"     constellation by +60 deg permutes the decoded sector index by "
          f"+1 -> {'PASS' if eq_pass else 'FAIL'} ({viol} violations)")
    print(f"   differential round-trip: exact data recovery at every k*60 deg "
          f"offset (zero noise) -> {'PASS' if rt_pass else 'FAIL'}")
    print("   [The 6-point sweep below is a didactic differential-6-PSK instance; "
          "the sector-index")
    print("    differential scheme generalizes to any hex constellation via the "
          "senary label s_6,")
    print("    with the caveat that a 60 deg rotation also permutes within-sector "
          "(shell) structure.]")

    rows = run_offset_sweep(args.dtheta, args.trials, args.ebn0, args.seed)
    print(f"\n{'dtheta(deg)':>11} {'coherent SER':>14} {'differential SER':>18}")
    for r in rows:
        print(f"{r['dtheta_deg']:>11.0f} {r['coherent_ser']:>14.3e} "
              f"{r['differential_ser']:>18.3e}")

    floor = next(r for r in rows if r["dtheta_deg"] == 0)
    at60 = next((r for r in rows if r["dtheta_deg"] == 60), None)
    print("\n  Observations:")
    print(f"   - zero-offset noise floor: coherent={floor['coherent_ser']:.3e}, "
          f"differential={floor['differential_ser']:.3e}")
    if floor["coherent_ser"] > 0:
        penalty = floor["differential_ser"] / max(floor["coherent_ser"], 1e-12)
        print(f"   - differential detection penalty at 0 deg: "
              f"~{penalty:.1f}x the coherent SER (error propagation in differences)")
    if at60 is not None:
        print(f"   - at 60 deg: coherent={at60['coherent_ser']:.3e} (sector slip), "
              f"differential={at60['differential_ser']:.3e} (back to ~noise floor)")

    csv_path = os.path.join(args.results_dir, "sim04_phase.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Persist the exact equivariance verification so the paper artifact set is
    # complete (not stdout-only): both checks must read PASS for C6.
    check_path = os.path.join(args.results_dir, "sim04_equivariance_check.csv")
    with open(check_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "result", "detail"])
        w.writerow(["decoder_equivariance_60deg", "PASS" if eq_pass else "FAIL",
                    f"{viol} violations over {npts} hex points"])
        w.writerow(["differential_roundtrip_k60deg", "PASS" if rt_pass else "FAIL",
                    "exact recovery at every k*60deg, zero noise"])
    print(f"\nWrote {csv_path}")
    print(f"Wrote {check_path}")

    # ---- C8: combined rotation + inversion (Z6 x Z2) differential codec -----
    # Additive and independent: uses its own RNG (args.t24_seed), so the two CSVs
    # written above remain byte-identical. The inversion bit is a discrete label
    # state, never a Euclidean operation (firewall).
    t24_viol, t24_rows = verify_t24_differential_invariance(
        args.t24_trials, args.t24_seed)
    t24_pass = (t24_viol == 0)
    print("\n[C8 verify] combined rotation + inversion (Z6 x Z2) differential codec:")
    print(f"   carries (sector in Z6, inversion-bit in Z2); transmits component-")
    print(f"   wise differences. Invariance over all 12 static (rotation,inversion)")
    print(f"   actions on a {args.t24_trials}-symbol stream -> "
          f"{'PASS' if t24_pass else 'FAIL'} ({t24_viol} violations)")
    print(f"   {'rot k':>6} {'inv m':>6} {'sector ok':>10} {'inv ok':>8} "
          f"{'viol':>6}")
    for r in t24_rows:
        print(f"   {r['rotation_k']:>6} {r['inversion_m']:>6} "
              f"{str(r['sector_recovered']):>10} {str(r['inversion_recovered']):>8} "
              f"{r['violations']:>6}")
    print("   [order-12 Z6 x Z2 = the rotation x inversion subgroup of the")
    print("    centrosymmetric hexagonal point group D_6h; stated without")
    print("    overclaiming the full 24-element group.]")

    t24_path = os.path.join(args.results_dir, "sim04_t24_check.csv")
    with open(t24_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rotation_k", "inversion_m", "sector_recovered",
                    "inversion_recovered", "violations"])
        for r in t24_rows:
            w.writerow([r["rotation_k"], r["inversion_m"],
                        int(r["sector_recovered"]), int(r["inversion_recovered"]),
                        r["violations"]])
    print(f"Wrote {t24_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
