#!/usr/bin/env python3
"""
simulation_06_phasepair_inversion_folded_decoder.py - Phase-Pair + Inversion-Folded Decoder (C1/C2 storage, C7 object)

Validates the phase-pair + inversion-folded demodulator for the Tri-Quarter
Framework (TQF) radial_dual_signal_processing subproject: an exact ML decoder
that stores only the fundamental-domain membership/label table yet returns
decisions that are bitwise-identical to exhaustive ML.

Claims validated
----------------
C1 (Exactness, preserved under folding): the folded decoder is bitwise-identical
    to exhaustive maximum-likelihood (ML) nearest-point decoding -- verified on a
    dense deterministic grid and on a Monte-Carlo channel stream (zero mismatches
    expected). The fold is a STORAGE/LABEL operation only; every Euclidean
    decision uses true distances (the inversion firewall: circle inversion is
    conformal, not isometric, and never touches a metric).
C2 (Resource reduction via folding): the membership/label table is reduced from
    the full per-shell representation to the fundamental domain. Two distinct,
    non-multiplied reductions are reported separately:
      * phase-pair (rotation) fold  -- store one representative per complete
        shell instead of all six sector points  -> exact 6x label-table fold;
      * phase-pair + inversion fold -- additionally store only the inner/boundary
        shells and regenerate the outer shells from their exact integer duals
        N -> r^4/N.
    These are storage facts, NOT a per-symbol latency claim: the radial-dual
    constellation is shell-complete with radial gaps, so its fast-path coverage
    is intentionally moderate; the headline here is exact ML at reduced storage.

Demonstration vehicle: the C7 radial-dual constellation (shell-complete, phase-
pair-uniform, inversion-paired about r^2 = 12; shells {3,4,9,12,16,36,48}, M=42),
built by ``tqf_hex_signal.build_radial_dual_constellation``. It is the unique
filled object here that is closed under BOTH the order-6 rotation C6 and the
radial inversion Z2, so both folds are mathematically exact on it.

Output: the two PASS/FAIL banners (exactness, internal consistency),
the storage-ablation table, the coverage-vs-SNR table, and the commutativity/
involution verification line.

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
from typing import List, Tuple

import numpy as np

import tqf_hex_signal as t


def verify_dense_grid(con: t.Constellation, ctx_pp, ctx_ppi,
                      extent: float, step: float) -> Tuple[int, int, int, float]:
    """Folded decode == exhaustive ML on a dense grid tiling the constellation.

    Returns (num_points, mismatches_pp, mismatches_ppi, fast_path_fraction_ppi).
    """
    g = np.arange(-extent, extent + step / 2, step)
    X, Y = np.meshgrid(g, g)
    grid = (X + 1j * Y).ravel()
    ml = t.decode_ml(grid, con.points_unit)
    idx_pp, _ = t.decode_hex_folded(grid, ctx_pp)
    idx_ppi, fast = t.decode_hex_folded(grid, ctx_ppi)
    return (grid.size,
            int(np.sum(idx_pp != ml)),
            int(np.sum(idx_ppi != ml)),
            float(np.mean(fast)))


def verify_monte_carlo(con: t.Constellation, ctx_pp, ctx_ppi,
                       trials: int, ebn0_db: float,
                       rng: np.random.Generator) -> Tuple[int, int, float]:
    """Folded decode == exhaustive ML on an AWGN stream. Returns
    (mismatches_pp, mismatches_ppi, fast_path_fraction_ppi)."""
    m = con.size
    tx = rng.integers(0, m, trials)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    ml = t.decode_ml(rx, con.points_unit)
    idx_pp, _ = t.decode_hex_folded(rx, ctx_pp)
    idx_ppi, fast = t.decode_hex_folded(rx, ctx_ppi)
    return (int(np.sum(idx_pp != ml)),
            int(np.sum(idx_ppi != ml)),
            float(np.mean(fast)))


def fast_path_fraction(con: t.Constellation, ctx, trials: int,
                       ebn0_db: float, rng: np.random.Generator) -> float:
    """Fraction of symbols whose nearest lattice point is a constellation point
    (the O(1) fast path) at a given Eb/N0."""
    m = con.size
    tx = rng.integers(0, m, trials)
    rx = t.awgn(con.points_unit[tx], ebn0_db, con.bits_per_symbol, rng)
    _idx, fast = t.decode_hex_folded(rx, ctx)
    return float(np.mean(fast))


def _time_build(con: t.Constellation, fold_inversion: bool, repeats: int) -> float:
    """Median precomputation (context-build) time in milliseconds."""
    t.make_folded_decode_context(con, fold_inversion=fold_inversion)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        t.make_folded_decode_context(con, fold_inversion=fold_inversion)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1e3


def _median_ns_per_symbol(decode_call, n_symbols: int, repeats: int) -> float:
    decode_call()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decode_call()
        t1 = time.perf_counter()
        samples.append((t1 - t0) / n_symbols)
    return float(np.median(samples)) * 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r_sq", type=int, default=12,
                    help="inversion radius^2 (12 is optimal for shells up to 60)")
    ap.add_argument("--max_norm_sq", type=int, default=60,
                    help="largest shell norm in the radial-dual constellation")
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--grid_extent", type=float, default=2.5)
    ap.add_argument("--grid_step", type=float, default=0.01)
    ap.add_argument("--ebn0", type=float, default=12.0,
                    help="Eb/N0 (dB) for the correctness Monte-Carlo stream")
    ap.add_argument("--ebn0_grid", type=float, nargs="+",
                    default=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
                    help="Eb/N0 grid (dB) for the fast-path coverage sweep")
    ap.add_argument("--throughput_batch", type=int, default=200_000)
    ap.add_argument("--timing_repeats", type=int, default=9)
    ap.add_argument("--coord_radius", type=int, default=8,
                    help="coordinate radius for the commutativity/involution check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print("=" * 84)
    print("SIMULATION 06 -- phase-pair + inversion-FOLDED decoder (exact ML, folded storage)")
    print(f"seed={args.seed}  trials={args.trials}  r^2={args.r_sq}  "
          f"max_norm_sq={args.max_norm_sq}")
    print("=" * 84)

    con = t.build_radial_dual_constellation(args.r_sq, args.max_norm_sq)
    prov_extra = {
        "constellation": "radial_dual",
        "M": con.size,
        "shell_norms": list(con.shell_norms),
        "inversion_r_sq": con.inversion_r_sq,
        "phase_pair_uniform": con.phase_pair_uniform,
        "inversion_paired": con.inversion_paired,
        "fundamental_domain_size": con.fundamental_domain_size,
        "scale_sq_exact": str(con.scale_sq_exact),
    }
    t.emit_provenance(args.results_dir, "sim06", args=args, extra=prov_extra)
    print(f"  C7 radial-dual constellation: M={con.size}, shells={list(con.shell_norms)}, "
          f"scale^2={con.scale_sq_exact}")
    print(f"  phase_pair_uniform={con.phase_pair_uniform}  "
          f"inversion_paired={con.inversion_paired}  "
          f"fundamental_domain_size={con.fundamental_domain_size}")

    ctx_pp = t.make_folded_decode_context(con, fold_inversion=False)   # rotation fold
    ctx_ppi = t.make_folded_decode_context(con, fold_inversion=True)   # + inversion fold

    # ---- C1: exactness of the folded decoder (grid + Monte-Carlo) -----------
    n_grid, mm_pp_g, mm_ppi_g, fast_g = verify_dense_grid(
        con, ctx_pp, ctx_ppi, args.grid_extent, args.grid_step)
    rng = np.random.default_rng(args.seed)
    mm_pp_mc, mm_ppi_mc, fast_mc = verify_monte_carlo(
        con, ctx_pp, ctx_ppi, args.trials, args.ebn0, rng)
    exact_pass = (mm_pp_g == 0 and mm_ppi_g == 0 and mm_pp_mc == 0 and mm_ppi_mc == 0)
    print(f"\n[C1 exactness] folded decode vs exhaustive ML (zero mismatches expected):")
    print(f"   dense grid ({n_grid} pts): phase-pair fold={mm_pp_g} mismatches, "
          f"phase-pair+inversion fold={mm_ppi_g} mismatches")
    print(f"   Monte-Carlo ({args.trials} sym @ {args.ebn0} dB): "
          f"phase-pair fold={mm_pp_mc}, phase-pair+inversion fold={mm_ppi_mc}")
    print(f"   -> {'PASS' if exact_pass else 'FAIL'} "
          f"(folding preserves bitwise-ML; inversion folds storage only)")

    # ---- Storage ablation (the two distinct, non-multiplied reductions) -----
    # The fold stores ONE representative per (fundamental) shell and regenerates
    # the six sector points by rotation, so the stored count is the shell count,
    # not the point count. Full per-point table = M labels.
    full_table = con.size                                   # one label per point
    pp_table = ctx_pp.stored_table_size                     # one per complete shell
    ppi_table = ctx_ppi.stored_table_size                   # inner/boundary shells
    pp_fold = full_table / pp_table                         # exact 6x (rotation)
    ppi_fold = full_table / ppi_table                       # 10.5x for r^2=12
    # Inversion-only marginal on the shell table: pp -> ppi.
    inv_marginal = pp_table / ppi_table
    build_pp_ms = _time_build(con, False, args.timing_repeats)
    build_ppi_ms = _time_build(con, True, args.timing_repeats)
    print(f"\n[C2 storage ablation] membership/label table size (smaller = better):")
    print(f"   {'representation':<34}{'stored':>8}{'vs full':>10}")
    print(f"   {'full per-point label table':<34}{full_table:>8}{'1.00x':>10}")
    print(f"   {'phase-pair (rotation) fold':<34}"
          f"{pp_table:>8}{pp_fold:>9.3f}x   (store 1 rep per shell, +5 by rotation)")
    print(f"   {'phase-pair + inversion fold':<34}"
          f"{ppi_table:>8}{ppi_fold:>9.3f}x   (store inner shells, outer via dual)")
    print(f"   inversion marginal on the shell table: {pp_table} -> {ppi_table} "
          f"shells ({inv_marginal:.3f}x). The self-dual boundary shell is not")
    print(f"   doubled, so the combined factor ({ppi_fold:.3f}x) is below the 12x ceiling.")
    print(f"   precomputation: phase-pair fold {build_pp_ms:.4f} ms, "
          f"phase-pair+inversion fold {build_ppi_ms:.4f} ms")
    print("   [Reductions reported SEPARATELY -- the exact 6x rotation fold and the")
    print("    inversion fold are distinct and are not multiplied into one number.]")

    # ---- C2 coverage: O(1) fast-path fraction vs SNR ------------------------
    print(f"\n[C2 coverage] fast-path fraction vs Eb/N0 (M={con.size}); "
          f"moderate by design (shell-complete -> radial gaps):")
    coverage_rows: List[tuple] = []
    rng_cov = np.random.default_rng(args.seed + 1)
    print(f"   {'Eb/N0(dB)':>9} {'fast-path':>10} {'fallback':>9}")
    for ebn0 in args.ebn0_grid:
        frac = fast_path_fraction(con, ctx_ppi, args.trials, ebn0, rng_cov)
        coverage_rows.append((con.size, ebn0, frac, 1.0 - frac))
        print(f"   {ebn0:>9.1f} {frac:>9.3f} {1.0 - frac:>9.3f}")

    # ---- Throughput (corroboration only; NOT the headline for this object) --
    rng_tp = np.random.default_rng(args.seed + 2)
    tx = rng_tp.integers(0, con.size, args.throughput_batch)
    rx = t.awgn(con.points_unit[tx], args.ebn0, con.bits_per_symbol, rng_tp)
    ns_folded = _median_ns_per_symbol(
        lambda: t.decode_hex_folded(rx, ctx_ppi), args.throughput_batch, args.timing_repeats)
    ns_ml = _median_ns_per_symbol(
        lambda: t.decode_ml(rx, con.points_unit), args.throughput_batch, args.timing_repeats)
    speedup = ns_ml / ns_folded if ns_folded > 0 else float("nan")
    print(f"\n[throughput] batched per-symbol decode @ {args.ebn0} dB "
          f"(corroboration, includes fold/mapping overhead):")
    print(f"   folded={ns_folded:.1f} ns/sym, exhaustive ML={ns_ml:.1f} ns/sym, "
          f"speedup={speedup:.2f}x")
    print("   NOTE: this object's value is EXACT ML at folded storage, not latency;")
    print("   its moderate fast-path coverage is expected (radial gaps).")

    # ---- Inversion involution + commutativity (cite: lattice paper Prop 4.15) -
    checked, inv_viol, comm_viol = t.verify_inversion_commutativity(
        args.r_sq, coord_radius=args.coord_radius)
    comm_pass = (inv_viol == 0 and comm_viol == 0)
    print(f"\n[label-fold soundness] exact inversion duality on labels "
          f"({checked} points, r^2={args.r_sq}):")
    print(f"   involution iota_r(iota_r(v)) == v: {inv_viol} violations; "
          f"sector(iota_r v) == sector(v): {comm_viol} violations -> "
          f"{'PASS' if comm_pass else 'FAIL'}")
    print("   (the sector-preservation commutativity is Prop. 4.15 of the lattice")
    print("    paper -- verified here empirically, cited rather than re-derived.)")

    overall = exact_pass and comm_pass and bool(ctx_ppi.full_shells == tuple(con.shell_norms))
    print(f"\n  C1/C2 RESULT: {'PASS' if overall else 'FAIL'} "
          f"(exact ML under both folds; label folds sound; storage reduced)")

    # ---- Write CSVs (new files; existing sim CSVs untouched) ----------------
    ablation_path = os.path.join(args.results_dir, "sim06_folded_ablation.csv")
    with open(ablation_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["representation", "stored_label_entries", "stored_shells",
                    "fold_vs_full", "precompute_ms"])
        w.writerow(["full_per_point", full_table, "", 1.0, ""])
        w.writerow(["phase_pair_fold", pp_table, pp_table, pp_fold, build_pp_ms])
        w.writerow(["phase_pair_inversion_fold", ppi_table, ppi_table,
                    ppi_fold, build_ppi_ms])

    coverage_path = os.path.join(args.results_dir, "sim06_coverage_vs_snr.csv")
    with open(coverage_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "ebn0_db", "fast_path_fraction", "fallback_fraction"])
        w.writerows(coverage_rows)

    summary_path = os.path.join(args.results_dir, "sim06_folded_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "shells", "fundamental_domain_size", "scale_sq_exact",
                    "grid_points", "grid_mismatch_pp", "grid_mismatch_ppi",
                    "mc_trials", "mc_mismatch_pp", "mc_mismatch_ppi",
                    "pp_fold", "ppi_fold", "inv_marginal",
                    "throughput_ns_folded", "throughput_ns_ml", "throughput_speedup",
                    "commutativity_violations", "involution_violations",
                    "exact_pass", "overall_pass"])
        w.writerow([con.size, "|".join(str(n) for n in con.shell_norms),
                    con.fundamental_domain_size, str(con.scale_sq_exact),
                    n_grid, mm_pp_g, mm_ppi_g, args.trials, mm_pp_mc, mm_ppi_mc,
                    pp_fold, ppi_fold, inv_marginal,
                    ns_folded, ns_ml, speedup, comm_viol, inv_viol,
                    int(exact_pass), int(overall)])

    print(f"\nWrote {ablation_path}")
    print(f"Wrote {coverage_path}")
    print(f"Wrote {summary_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
