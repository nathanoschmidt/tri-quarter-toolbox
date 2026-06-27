#!/usr/bin/env python3
"""
simulation_07_radial_dual_constellation.py - Phase-Pair-Uniform, Inversion-Paired Radial-Dual Constellation (C7)

Verifies the exact structural properties of the C7 radial-dual hexagonal
constellation for the Tri-Quarter Framework (TQF) radial_dual_signal_processing
subproject.

Claim validated
---------------
C7 (Radial-dual constellation): there exists a filled hexagonal constellation
    that is simultaneously (i) shell-complete (every included Eisenstein-norm
    shell contains all of its lattice points, giving exact order-6 rotation
    symmetry C6), (ii) phase-pair-uniform (every complete shell has identical
    angular-sector occupancy, k points per sector), and (iii) inversion-paired
    (closed under the exact circle inversion iota_r about r^2 = 12, which maps
    each shell of norm N to its integer-dual shell of norm r^4/N = 144/N). It is
    therefore invariant under the full order-12 rotation x inversion group
    C6 x Z2 -- the operationally relevant rotation-and-inversion subgroup of the
    centrosymmetric hexagonal point group D_6h. Because 6 does not divide any
    power of two, a shell-complete object and a power-of-two filled object are
    mutually exclusive; this construction takes shell-completeness (Option B),
    which is the only choice that places the full C6 x Z2 structure on a single
    exact object with uniform sector occupancy.

All structural facts are checked in exact integer/rational arithmetic (==, not a
tolerance). Inversion is a label/structure operation here -- a Z2 involution on
shells and a same-sector point permutation -- never a Euclidean operation (the
firewall: iota_r is conformal, not isometric).

Output: the structure PASS/FAIL banner, the integer-dual shell-pair
table, and the per-shell occupancy table.

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
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

import tqf_hex_signal as t


def _shell_occupancy(con: t.Constellation) -> List[Tuple[int, int, List[int], int]]:
    """Per-shell (norm, |shell|, sorted sector list, per-sector count k)."""
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    norms = a * a + a * b + b * b
    sectors = t.phase_pair_sector_array(con.ab)
    rows: List[Tuple[int, int, List[int], int]] = []
    for N in con.shell_norms:
        mask = norms == N
        sec = sorted(set(int(s) for s in sectors[mask]))
        # per-sector count: must be identical across all six sectors
        per = [int(np.sum(sectors[mask] == s)) for s in range(6)]
        k = per[0] if len(set(per)) == 1 else -1   # -1 flags non-uniform occupancy
        rows.append((int(N), int(np.sum(mask)), sec, k))
    return rows


def _verify_inversion_permutation(con: t.Constellation
                                  ) -> Tuple[bool, bool, int]:
    """Confirm inversion induces a same-sector point permutation that is an
    involution, fixing the self-dual boundary shell pointwise.

    Returns (is_same_sector_permutation, is_involution, num_boundary_fixed).
    """
    dual = con.inversion_dual_index
    if dual is None:
        return False, False, 0
    dual = np.asarray(dual, dtype=np.int64)
    n = con.size
    # permutation: dual is a bijection of {0..n-1}
    is_perm = bool(np.array_equal(np.sort(dual), np.arange(n)))
    # involution: applying twice is the identity
    is_invol = bool(np.array_equal(dual[dual], np.arange(n)))
    # same-sector: each point and its dual share the phase-pair sector
    sectors = t.phase_pair_sector_array(con.ab)
    same_sector = bool(np.array_equal(sectors, sectors[dual]))
    # boundary fixed: points on the inversion circle (norm == r^2) are fixed
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    norms = a * a + a * b + b * b
    on_circle = norms == con.inversion_r_sq
    boundary_fixed = int(np.sum(dual[on_circle] == np.where(on_circle)[0]))
    return (is_perm and same_sector), is_invol, boundary_fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r_sq", type=int, default=12,
                    help="inversion radius^2 (12 is optimal for shells up to 60)")
    ap.add_argument("--max_norm_sq", type=int, default=60,
                    help="largest shell norm in the radial-dual constellation")
    ap.add_argument("--seed", type=int, default=42)  # determinism only; unused
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    print("=" * 84)
    print("SIMULATION 07 -- radial-dual constellation structure (phase-pair + inversion, C7)")
    print(f"r^2={args.r_sq}  max_norm_sq={args.max_norm_sq}  "
          f"(integer-dual shells satisfy N * N' = r^4 = {args.r_sq ** 2})")
    print("=" * 84)

    con = t.build_radial_dual_constellation(args.r_sq, args.max_norm_sq)

    # ---- integer-dual shell pairs (N, r^4/N), both within max_norm_sq -------
    pairs = t.radial_dual_shell_pairs(args.r_sq, args.max_norm_sq)
    r4 = args.r_sq * args.r_sq

    prov_extra = {
        "constellation": "radial_dual",
        "M": con.size,
        "shell_norms": list(con.shell_norms),
        "inversion_r_sq": con.inversion_r_sq,
        "phase_pair_uniform": con.phase_pair_uniform,
        "inversion_paired": con.inversion_paired,
        "fundamental_domain_size": con.fundamental_domain_size,
        "scale_sq_exact": str(con.scale_sq_exact),
        "shell_pairs": [list(p) for p in pairs],
    }
    t.emit_provenance(args.results_dir, "sim07", args=args, extra=prov_extra)

    print(f"\n[constellation] M={con.size}, shells={list(con.shell_norms)}, "
          f"scale^2={con.scale_sq_exact} (exact)")
    print(f"  fundamental_domain_size (sector-0 inner+boundary) = "
          f"{con.fundamental_domain_size}")

    print(f"\n[integer-dual shell pairs] N * N' = r^4 = {r4}, both shells <= "
          f"{args.max_norm_sq}:")
    print(f"   {'N':>6} {'N_dual':>8}  note")
    pair_rows: List[tuple] = []
    for (lo, hi) in pairs:
        note = "self-dual (on inversion circle)" if lo == hi else "inner <-> outer"
        print(f"   {lo:>6} {hi:>8}  {note}")
        pair_rows.append((lo, hi, "self_dual" if lo == hi else "inner_outer"))

    # ---- per-shell occupancy: phase-pair uniformity ------------------------
    occ = _shell_occupancy(con)
    pp_uniform = all(k == occ[0][3] for (_, _, _, k) in occ) and all(
        k > 0 for (_, _, _, k) in occ) and all(
        sec == [0, 1, 2, 3, 4, 5] for (_, _, sec, _) in occ)
    print(f"\n[phase-pair uniformity] each complete shell has k points per sector "
          f"(all six sectors equal):")
    print(f"   {'N':>6} {'|shell|':>8} {'k/sector':>9} {'sectors':>16}")
    occ_rows: List[tuple] = []
    for (N, size, sec, k) in occ:
        print(f"   {N:>6} {size:>8} {k:>9} {str(sec):>16}")
        occ_rows.append((N, size, k, "".join(str(s) for s in sec)))

    # ---- inversion pairing: involution + same-sector permutation -----------
    perm_ok, invol_ok, boundary_fixed = _verify_inversion_permutation(con)
    print(f"\n[inversion pairing] iota_r as an exact Z2 structure on the labels:")
    print(f"   induces a SAME-SECTOR point permutation: {perm_ok}")
    print(f"   permutation is an involution (iota_r^2 = id): {invol_ok}")
    print(f"   self-dual boundary shell (norm {args.r_sq}) fixed pointwise: "
          f"{boundary_fixed} points")

    # ---- exact involution + commutativity on raw labels (Prop 4.15) --------
    checked, inv_viol, comm_viol = t.verify_inversion_commutativity(
        args.r_sq, coord_radius=8)
    print(f"\n[exact label duality] over {checked} lattice points (r^2={args.r_sq}):")
    print(f"   involution iota_r(iota_r(v)) == v: {inv_viol} violations")
    print(f"   sector(iota_r v) == sector(v)   : {comm_viol} violations "
          f"(Prop. 4.15 of the lattice paper, verified not re-derived)")

    structure_pass = bool(
        con.phase_pair_uniform and con.inversion_paired and pp_uniform and
        perm_ok and invol_ok and inv_viol == 0 and comm_viol == 0 and
        boundary_fixed == 6)
    print(f"\n  C7 RESULT: {'PASS' if structure_pass else 'FAIL'} "
          f"(shell-complete C6 x Z2 object; phase-pair-uniform; inversion-paired; "
          f"exact)")
    print("  [order-12 C6 x Z2 = the rotation x inversion subgroup of D_6h; stated")
    print("   as such, without overclaiming the full 24-element point group.]")

    # ---- write CSVs (new files; existing sim CSVs untouched) ---------------
    pairs_path = os.path.join(args.results_dir, "sim07_shell_pairs.csv")
    with open(pairs_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "N_dual", "kind", "r4"])
        for (lo, hi, kind) in pair_rows:
            w.writerow([lo, hi, kind, r4])

    occ_path = os.path.join(args.results_dir, "sim07_occupancy.csv")
    with open(occ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "shell_size", "k_per_sector", "sectors"])
        w.writerows(occ_rows)

    summary_path = os.path.join(args.results_dir, "sim07_structure.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "shells", "inversion_r_sq", "scale_sq_exact",
                    "fundamental_domain_size", "num_shell_pairs",
                    "phase_pair_uniform", "inversion_paired",
                    "same_sector_permutation", "involution",
                    "boundary_fixed", "commutativity_violations",
                    "involution_violations", "structure_pass"])
        w.writerow([con.size, "|".join(str(n) for n in con.shell_norms),
                    con.inversion_r_sq, str(con.scale_sq_exact),
                    con.fundamental_domain_size, len(pairs),
                    int(bool(con.phase_pair_uniform)),
                    int(bool(con.inversion_paired)),
                    int(perm_ok), int(invol_ok), boundary_fixed,
                    comm_viol, inv_viol, int(structure_pass)])

    print(f"\nWrote {pairs_path}")
    print(f"Wrote {occ_path}")
    print(f"Wrote {summary_path}")
    print("=" * 84)


if __name__ == "__main__":
    main()
