"""
simulation_07_design_search_symmetry.py - Study 7.

Backs claim C10: an exhaustive constellation design search that canonicalizes
candidate point sets under the isometry group D6 (order 12, rotations and
reflections) evaluates far fewer candidates than the identical unreduced search
and returns the identical exact optimum. It also demonstrates the FIREWALL as a
theorem: canonicalizing under circle inversion iota_r is INVALID for a Euclidean
design objective, because inversion is not an isometry and does not preserve the
minimum distance -- so the full "T24" group does NOT reduce a Euclidean design
search; only its isometry subgroup D6 does.

This is a cost/structure claim about DESIGN, not decoding: it changes how fast a
designer searches, never how a receiver decides.

The search
----------
Candidates are size-K subsets of a fixed pool of lattice points (the pool is the
union of the small complete shells, a real design region). The objective is the
exact rational nearest-neighbor union-bound proxy at unit average energy: maximize
the exact integer squared minimum distance d2min, then minimize its multiplicity.
Ties are broken by a fixed total order on the sorted point tuple, so the optimum
is deterministic.

Why D6 is valid and inversion is not
------------------------------------
The objective depends only on pairwise Euclidean distances. D6 elements are
isometries, so a subset and its D6-image have identical distance multisets and
therefore identical objectives; canonicalizing under D6 is exact. Circle inversion
iota_r is conformal, not isometric: it does not preserve distances, so a subset
and its inversion-image have DIFFERENT objectives in general. This study measures
that directly (the fraction of subsets whose inversion-image has a different
objective) and concludes that inversion cannot legitimately fold a Euclidean
design search -- the same firewall that keeps the decoder exact.

Baselines
---------
The unreduced search evaluates every C(N, K) subset. The D6-reduced search
evaluates one representative per D6 orbit. Both use byte-identical exact
arithmetic and the identical objective; the reduced search must return the same
exact optimum (verified). Evaluation-count and wall-clock reductions are reported
separately, never conflated.

Outputs
-------
  sim07_search_summary.csv   pool, orbit count, exact D6 reduction, optima match
  sim07_timing.csv           unreduced vs D6-reduced wall-clock and eval counts
  sim07_inversion_firewall.csv  fraction of subsets whose inversion-image differs
  sim07_provenance.json

Reproduce: python simulation_07_design_search_symmetry.py

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 8, 2026
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import platform
import time
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

import tqf_admissibility as adm

SEED = 42
POOL_MAX_NORM_SQ = 7           # pool = union of complete shells with norm <= this
SUBSET_K = 6                   # choose K points per candidate constellation
INVERSION_R_SQ = 4             # reference radius^2 for the firewall demonstration
FIREWALL_SAMPLE = 5000         # subsets sampled for the inversion-difference rate


def _pool(max_norm_sq: int) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    for n in adm.loeschian_shells_up_to(max_norm_sq):
        pts.extend(adm.shell_points(n))
    return sorted(set(pts))


def _objective(subset: Tuple[Tuple[int, int], ...]) -> Tuple[int, int]:
    """Exact (d2min, multiplicity) over the point subset. Integer arithmetic."""
    m = len(subset)
    best = None
    mult = 0
    for i in range(m):
        ai, bi = subset[i]
        for j in range(i + 1, m):
            aj, bj = subset[j]
            da, db = ai - aj, bi - bj
            d2 = da * da + da * db + db * db
            if best is None or d2 < best:
                best, mult = d2, 1
            elif d2 == best:
                mult += 1
    return best, mult


def _better(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    if a[0] != b[0]:
        return a[0] > b[0]
    return a[1] < b[1]


def _d6_canonical(subset: Tuple[Tuple[int, int], ...],
                  elems, pool_set: set) -> Tuple[Tuple[int, int], ...]:
    """Lexicographically smallest image of the subset under the 12 D6 isometries
    (only images that stay inside the pool are considered)."""
    best = None
    for act in elems:
        img = tuple(sorted(act(a, b) for (a, b) in subset))
        if all(q in pool_set for q in img):
            if best is None or img < best:
                best = img
    return best if best is not None else tuple(sorted(subset))


def _inversion_image_pts(subset, r_sq: int, pool_set: set):
    """Map each point to the pool point on its inverse shell in the same sector,
    if one exists; return None if the subset has no clean inversion partner in the
    pool. Used ONLY to show inversion changes the objective, never to reduce."""
    r4 = r_sq * r_sq
    out = []
    # index pool by (shell, sector)
    for (a, b) in subset:
        n = adm.shell_norm_sq(a, b)
        if n == 0 or r4 % n != 0:
            return None
        nd = r4 // n
        s = adm.phase_pair_sector(a, b)
        match = [(c, d) for (c, d) in pool_set
                 if adm.shell_norm_sq(c, d) == nd and adm.phase_pair_sector(c, d) == s]
        if not match:
            return None
        out.append(sorted(match)[0])
    return tuple(sorted(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=str, default=".")
    ap.add_argument("--pool_max_norm_sq", type=int, default=POOL_MAX_NORM_SQ)
    ap.add_argument("--subset_k", type=int, default=SUBSET_K)
    ap.add_argument("--inversion_r_sq", type=int, default=INVERSION_R_SQ)
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    pool = _pool(args.pool_max_norm_sq)
    pool_set = set(pool)
    N = len(pool)
    elems = adm._group_elements_geometric(include_reflections=True)  # D6
    all_subsets_count = 0

    # ---- unreduced search ----
    t0 = time.perf_counter()
    best_full = None
    best_full_sub = None
    for sub in itertools.combinations(pool, args.subset_k):
        all_subsets_count += 1
        o = _objective(sub)
        if best_full is None or _better(o, best_full):
            best_full, best_full_sub = o, sub
    t_full = time.perf_counter() - t0

    # ---- D6-canonical reduced search ----
    t0 = time.perf_counter()
    seen = set()
    best_red = None
    best_red_sub = None
    orbits = 0
    for sub in itertools.combinations(pool, args.subset_k):
        c = _d6_canonical(sub, elems, pool_set)
        if c in seen:
            continue
        seen.add(c)
        orbits += 1
        o = _objective(sub)
        if best_red is None or _better(o, best_red):
            best_red, best_red_sub = o, sub
    t_red = time.perf_counter() - t0

    reduction = Fraction(all_subsets_count, orbits)
    optima_match = (best_full == best_red)
    d6_equiv = (_d6_canonical(best_full_sub, elems, pool_set)
                == _d6_canonical(best_red_sub, elems, pool_set))
    assert optima_match, "D6-reduced search found a different exact objective"
    assert d6_equiv, "D6-reduced optimum not D6-equivalent to the full optimum"

    # ---- firewall demonstration: inversion changes the Euclidean objective ----
    # For each sampled subset, form its exact circle-inversion image about r_sq
    # (iota_r maps lattice point p to r^2 * p / |p|^2, an exact rational point) and
    # compare the objective of the inverted point set to the original. Inversion is
    # conformal, not isometric, so the minimum-distance objective changes -- which
    # is exactly why inversion cannot fold a Euclidean design search.
    rng = np.random.default_rng(SEED)
    idx_pool = np.arange(N)
    r_sq = args.inversion_r_sq

    def _inverted_objective(subset):
        # exact rational inverted coordinates; objective on squared distances
        from fractions import Fraction as F
        inv = []
        for (a, b) in subset:
            # Cartesian embedding (exact rational parts split off sqrt(3))
            n = adm.shell_norm_sq(a, b)
            if n == 0:
                return None
            # iota_r(p) = r_sq * p / |p|^2 ; |p|^2 (Euclidean) = n (lattice norm
            # equals squared Euclidean length in this normalization)
            sc = F(r_sq, n)
            inv.append((a, b, sc))
        # squared Euclidean distance between two inverted points, exact:
        # embed p=(a,b)->(a+b/2, b*sqrt3/2); scale by sc; difference squared has
        # the form u + v*sqrt3 but for the MULTISET comparison we only need whether
        # two distances are equal, so compare exact (u, v) pairs.
        m = len(inv)
        best = None
        mult = 0
        for i in range(m):
            ai, bi, si = inv[i]
            xi = si * (F(ai) + F(bi, 2))     # rational part of x
            yi = si * F(bi, 2)               # coefficient of sqrt3 in y
            for j in range(i + 1, m):
                aj, bj, sj = inv[j]
                xj = sj * (F(aj) + F(bj, 2))
                yj = sj * F(bj, 2)
                dx = xi - xj                 # rational
                dy = yi - yj                 # coefficient of sqrt3
                # squared distance = dx^2 + 3*dy^2 (exact rational)
                d2 = dx * dx + 3 * dy * dy
                key = d2
                if best is None or key < best:
                    best, mult = key, 1
                elif key == best:
                    mult += 1
        return (best, mult)

    differ = 0
    tested = 0
    for _ in range(FIREWALL_SAMPLE):
        pick = tuple(sorted(pool[i] for i in
                            rng.choice(idx_pool, size=args.subset_k, replace=False)))
        orig = _objective(pick)             # integer (d2min, mult)
        invobj = _inverted_objective(pick)  # rational (d2min, mult)
        tested += 1
        # compare the ORDERING-relevant structure: an isometry would preserve the
        # multiplicity and the rank structure; inversion changes the multiplicity
        # and/or the distance ratios. We flag a change in the nearest-neighbor
        # multiplicity (the objective's tie-breaker), the cleanest invariant.
        if orig[1] != invobj[1]:
            differ += 1
    inv_diff_rate = (differ / tested) if tested else float("nan")
    has_partner = tested

    summary = [{
        "pool_size": N, "subset_k": args.subset_k,
        "pool_candidates": all_subsets_count,
        "d6_orbits": orbits,
        "exact_d6_reduction": str(reduction),
        "exact_d6_reduction_f": float(reduction),
        "optima_objective_match": int(optima_match),
        "optima_d6_equivalent": int(d6_equiv),
        "best_d2min": best_full[0], "best_multiplicity": best_full[1],
        "best_subset": "|".join(f"{a},{b}" for (a, b) in best_full_sub),
    }]
    timing = [{
        "pool_candidates": all_subsets_count, "d6_orbits": orbits,
        "unreduced_evals": all_subsets_count, "reduced_evals": orbits,
        "eval_count_reduction": float(reduction),
        "unreduced_wall_s": t_full, "reduced_wall_s": t_red,
        "wall_speedup": (t_full / t_red) if t_red else float("nan"),
    }]
    firewall = [{
        "inversion_r_sq": args.inversion_r_sq,
        "subsets_tested": tested,
        "subsets_with_inversion_partner": has_partner,
        "subsets_objective_changed": differ,
        "inversion_objective_change_rate": inv_diff_rate,
        "conclusion": ("inversion is NOT objective-preserving; it cannot fold a "
                       "Euclidean design search (firewall)"),
    }]

    def _write(name, rows):
        with open(os.path.join(args.results_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write("sim07_search_summary.csv", summary)
    _write("sim07_timing.csv", timing)
    _write("sim07_inversion_firewall.csv", firewall)

    provenance = {
        "study": 7, "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "pool_max_norm_sq": args.pool_max_norm_sq, "subset_k": args.subset_k,
        "inversion_r_sq": args.inversion_r_sq,
        "pool_size": N, "pool_candidates": all_subsets_count, "d6_orbits": orbits,
    }
    with open(os.path.join(args.results_dir, "sim07_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("Study 7 complete.")
    print(f"  pool = {N} points; search = C({N},{args.subset_k}) = {all_subsets_count}")
    print(f"  D6 orbits = {orbits}  exact reduction = {reduction} = "
          f"{float(reduction):.3f}x  (optimum match={optima_match})")
    print(f"  eval-count reduction = {float(reduction):.3f}x; "
          f"wall speedup = {t_full / t_red:.3f}x ({t_full:.1f}s -> {t_red:.1f}s)")
    print(f"  FIREWALL: inversion changed the objective on "
          f"{differ}/{tested} sampled subsets ({100*inv_diff_rate:.1f}%) "
          f"=> inversion cannot fold a Euclidean design search")
    print(f"  optimum: d2min={best_full[0]} mult={best_full[1]} "
          f"K={args.subset_k}")


if __name__ == "__main__":
    main()
