"""
tqf_admissibility.py - Radial dual family enumeration and exact fold-factor audit.

This module answers the two structural questions the design-search and
fold-at-scale studies depend on, both in exact integer arithmetic:

  1. Which radial dual constellations exist? A radial dual constellation about
     inversion radius r (r^2 = r_sq) is the union of complete lattice shells that
     is closed under circle inversion: every shell norm N present has its integer
     dual r^4 / N present too. Enumerating the admissible (r_sq, shell-set) pairs
     up to an energy bound, and their resulting orders M, tells us how large the
     family gets -- the go/no-go gate for any "fold at scale" claim.

  2. By how much does symmetry fold the work? For a set of lattice points closed
     under a symmetry group G, evaluating a G-invariant quantity on one
     representative per orbit costs |orbits| instead of |points|. The reduction
     factor is |points| / |orbits|, which by Burnside's lemma equals the average
     number of points fixed by a group element. We compute this exactly for the
     rotation group Z6, the full isometry group D6 (rotations and reflections),
     and the label-domain group Z6 x Z2 and D6 x Z2 (adjoining circle inversion),
     counting fixed points honestly rather than assuming the round group order.

Neither computation uses floating point: shells are integer Eisenstein norms,
group actions are integer coordinate maps, and inversion acts on integer shell
labels via N -> r^4 / N.

Run as a script to print the family table and the fold-factor table and to write
``mark4_admissibility.json`` for the downstream studies to consume.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Exact lattice primitives (self-contained; no signal-library dependency)
# ---------------------------------------------------------------------------


def shell_norm_sq(a: int, b: int) -> int:
    """Integer Eisenstein norm a^2 + a*b + b^2 (squared Euclidean length)."""
    return a * a + a * b + b * b


def rotate60(a: int, b: int) -> Tuple[int, int]:
    """Order-6 rotation R_{pi/3}(a, b) = (-b, a + b); applied six times it is the identity."""
    return (-b, a + b)


def reflect(a: int, b: int) -> Tuple[int, int]:
    """A lattice reflection: (a, b) -> (b, a).

    Swapping oblique coordinates mirrors the point across the omega0+omega1
    diagonal; it is an isometry of the triangular lattice and, together with the
    order-6 rotation, generates the full dihedral group D6 of order 12.
    """
    return (b, a)


def primary_directions() -> Tuple[Tuple[int, int], ...]:
    """The six primary ray directions R^k(1, 0), k = 0..5, as integer pairs."""
    dirs = []
    p = (1, 0)
    for _ in range(6):
        dirs.append(p)
        p = rotate60(*p)
    return tuple(dirs)


def _det(ua: int, ub: int, va: int, vb: int) -> int:
    return ua * vb - va * ub


def phase_pair_sector(a: int, b: int) -> int:
    """Exact phase-pair sector in {0..5}; origin -> -1. Integer cross products."""
    if a == 0 and b == 0:
        return -1
    dirs = primary_directions()
    for k in range(6):
        da, db = dirs[k]
        ea, eb = dirs[(k + 1) % 6]
        if _det(da, db, a, b) >= 0 and _det(a, b, ea, eb) > 0:
            return k
    return -1


def loeschian_shells_up_to(max_norm_sq: int) -> List[int]:
    """Sorted positive Eisenstein norms realized by points with norm <= bound."""
    radius = int((4 * max_norm_sq / 3) ** 0.5) + 1
    norms = set()
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            n = shell_norm_sq(a, b)
            if 0 < n <= max_norm_sq:
                norms.add(n)
    return sorted(norms)


def _coord_bound(n_sq: int) -> int:
    """Max possible |a| or |b| on shell a^2+a*b+b^2 = n_sq.

    Minimizing a^2+a*b+b^2 over b for fixed a gives 3a^2/4, so |a| <= sqrt(4N/3);
    the same bound holds for |b| by symmetry. Add 1 for the integer ceiling.
    """
    return int((4 * n_sq / 3) ** 0.5) + 1


def shell_points(n_sq: int) -> List[Tuple[int, int]]:
    """All lattice points on the shell of squared norm ``n_sq``."""
    radius = _coord_bound(n_sq)
    return [(a, b)
            for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1)
            if shell_norm_sq(a, b) == n_sq]


# ---------------------------------------------------------------------------
# Radial dual family enumeration
# ---------------------------------------------------------------------------


@dataclass
class RadialDualCandidate:
    """One admissible radial dual constellation."""
    r_sq: int
    max_norm_sq: int
    shells: Tuple[int, ...]           # complete shell norms present
    shell_pairs: Tuple[Tuple[int, int], ...]   # (N, N_dual), N <= N_dual
    self_dual_shell: int | None       # N == r_sq, on the inversion circle, or None
    M: int                            # total points (sum of shell sizes)
    k_per_sector: Tuple[int, ...]     # per-shell points-per-sector (uniformity)
    phase_pair_uniform: bool          # every shell has equal per-sector occupancy


def shell_size(n_sq: int) -> int:
    """Number of lattice points on shell ``n_sq`` (always a multiple of 6)."""
    return len(shell_points(n_sq))


def radial_dual_shell_pairs(r_sq: int, max_norm_sq: int) -> List[Tuple[int, int]]:
    """Inversion-dual complete-shell pairs (N, N_dual) with N <= N_dual, both
    Eisenstein norms <= max_norm_sq. The self-dual shell N == r_sq appears as
    (r_sq, r_sq)."""
    shells = set(loeschian_shells_up_to(max_norm_sq))
    r4 = r_sq * r_sq
    out: List[Tuple[int, int]] = []
    for n in sorted(shells):
        if r4 % n != 0:
            continue
        nd = r4 // n
        if nd in shells and n <= nd:
            out.append((n, nd))
    return out


def build_candidate(r_sq: int, max_norm_sq: int) -> RadialDualCandidate | None:
    """Assemble the admissible radial dual constellation for (r_sq, max_norm_sq),
    or None if no inversion-dual shell pair exists within the bound."""
    pairs = radial_dual_shell_pairs(r_sq, max_norm_sq)
    if not pairs:
        return None
    shells = tuple(sorted({n for pair in pairs for n in pair}))
    self_dual = r_sq if any(n == nd == r_sq for (n, nd) in pairs) else None

    k_list: List[int] = []
    uniform = True
    total = 0
    for n in shells:
        pts = shell_points(n)
        total += len(pts)
        per_sector: Dict[int, int] = {k: 0 for k in range(6)}
        for (a, b) in pts:
            per_sector[phase_pair_sector(a, b)] += 1
        vals = set(per_sector.values())
        k_list.append(next(iter(vals)) if len(vals) == 1 else -1)
        if len(vals) != 1:
            uniform = False

    return RadialDualCandidate(
        r_sq=r_sq, max_norm_sq=max_norm_sq, shells=shells,
        shell_pairs=tuple(pairs), self_dual_shell=self_dual, M=total,
        k_per_sector=tuple(k_list), phase_pair_uniform=uniform,
    )


def enumerate_family(r_sq_values: Sequence[int],
                     max_norm_sq: int) -> List[RadialDualCandidate]:
    """Enumerate admissible radial dual constellations over a range of radii.

    For each r_sq we take the richest admissible shell set within the energy
    bound (all inversion-dual pairs up to max_norm_sq). Returns the candidates
    sorted by order M, deduplicated by shell set so the family is a clean size
    ladder for the fold-at-scale sweep.
    """
    seen: Dict[Tuple[int, ...], RadialDualCandidate] = {}
    for r_sq in r_sq_values:
        cand = build_candidate(r_sq, max_norm_sq)
        if cand is None:
            continue
        key = cand.shells
        # Prefer the smallest r_sq that realizes a given shell set (arbitrary but
        # deterministic); the shell set is what defines the constellation.
        if key not in seen or r_sq < seen[key].r_sq:
            seen[key] = cand
    return sorted(seen.values(), key=lambda c: (c.M, c.r_sq))


# ---------------------------------------------------------------------------
# Exact fold factors by Burnside's lemma
# ---------------------------------------------------------------------------


def _group_elements_geometric(include_reflections: bool):
    """Return the geometric group elements as integer coordinate maps.

    Each element is a callable (a, b) -> (a', b'). Z6 is the six rotations;
    adding reflections gives all twelve elements of D6.
    """
    elems = []
    # rotations R^k
    for k in range(6):
        def rot(a, b, k=k):
            for _ in range(k):
                a, b = rotate60(a, b)
            return (a, b)
        elems.append(rot)
    if include_reflections:
        for k in range(6):
            def ref(a, b, k=k):
                a, b = reflect(a, b)
                for _ in range(k):
                    a, b = rotate60(a, b)
                return (a, b)
            elems.append(ref)
    return elems


def burnside_geometric_fold(points: Sequence[Tuple[int, int]],
                            include_reflections: bool) -> Fraction:
    """Exact fold factor |points| / |orbits| under Z6 (or D6) on a point set.

    By the orbit-counting theorem, |orbits| = (1/|G|) * sum over g of |Fix(g)|.
    Everything here is integer; the returned reduction factor is an exact
    Fraction. The point set must be closed under the group (radial dual shells
    are). Points are matched by exact integer coordinates.
    """
    pset = set(points)
    elems = _group_elements_geometric(include_reflections)
    g = len(elems)
    fixed_total = 0
    for act in elems:
        for p in pset:
            if act(*p) == p:
                fixed_total += 1
    # |orbits| = fixed_total / g  (an integer iff the set is G-closed)
    num_orbits = Fraction(fixed_total, g)
    if num_orbits.denominator != 1:
        raise ValueError(
            "point set is not closed under the group (non-integral orbit "
            "count); check shell completeness")
    return Fraction(len(pset), int(num_orbits))


def burnside_label_fold(shells: Sequence[int], r_sq: int,
                        include_reflections: bool) -> Fraction:
    """Exact label-domain fold adjoining circle inversion (the Z2 factor).

    The label domain is the set of (shell, sector) cells. The geometric group
    (Z6 or D6) acts on the sector; inversion iota_r acts by swapping shell N with
    its dual r^4 / N while fixing the sector. We build the full label-cell set,
    apply Burnside over the direct-product group (order 12 for Z6 x Z2, 24 for
    D6 x Z2), and return |cells| / |orbits| exactly.

    Reflections act on the sector as a dihedral flip; we realize the whole action
    concretely on representative lattice points (one per (shell, sector) cell) so
    that the sector image is computed by the exact phase-pair test, never assumed.
    """
    # Build one representative lattice point per (shell, sector) cell.
    reps: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for n in shells:
        for (a, b) in shell_points(n):
            s = phase_pair_sector(a, b)
            reps.setdefault((n, s), (a, b))
    cells = list(reps.keys())
    cell_set = set(cells)

    def dual_shell(n: int) -> int:
        r4 = r_sq * r_sq
        assert r4 % n == 0, "shell has no integer inversion dual"
        return r4 // n

    geo = _group_elements_geometric(include_reflections)

    def cell_image_geo(cell, act):
        n, s = cell
        a, b = reps[cell]
        ia, ib = act(a, b)
        return (shell_norm_sq(ia, ib), phase_pair_sector(ia, ib))

    def cell_image_inv(cell):
        n, s = cell
        return (dual_shell(n), s)   # inversion preserves sector, swaps shell

    # Product group: (geometric element) x (inversion^{0 or 1}).
    fixed_total = 0
    g_order = len(geo) * 2
    for act in geo:
        for inv_flag in (False, True):
            for cell in cells:
                img = cell_image_geo(cell, act)
                if inv_flag:
                    img = cell_image_inv(img)
                if img == cell and img in cell_set:
                    fixed_total += 1
    num_orbits = Fraction(fixed_total, g_order)
    assert num_orbits.denominator == 1, "label orbit count must be integral"
    return Fraction(len(cells), int(num_orbits))


@dataclass
class FoldAudit:
    """Exact fold factors for one radial dual constellation.

    Field names keep the historical c6/d6 spelling for continuity with the
    committed result CSVs; the paper writes the rotation group as Z6.
    """
    r_sq: int
    M: int
    shells: Tuple[int, ...]
    geo_c6: str          # Euclidean point fold under Z6 (exact Fraction as str)
    geo_d6: str          # Euclidean point fold under D6 (rotations + reflections)
    label_c6_z2: str     # label fold under Z6 x Z2 (10.5x at M=42)
    label_d6_z2: str     # label fold under D6 x Z2 (the full T24 label ceiling)
    geo_c6_f: float
    geo_d6_f: float
    label_c6_z2_f: float
    label_d6_z2_f: float


def audit_candidate(cand: RadialDualCandidate) -> FoldAudit:
    """Compute all four exact fold factors for a radial dual candidate."""
    pts: List[Tuple[int, int]] = []
    for n in cand.shells:
        pts.extend(shell_points(n))
    geo_c6 = burnside_geometric_fold(pts, include_reflections=False)
    geo_d6 = burnside_geometric_fold(pts, include_reflections=True)
    lab_c6 = burnside_label_fold(cand.shells, cand.r_sq, include_reflections=False)
    lab_d6 = burnside_label_fold(cand.shells, cand.r_sq, include_reflections=True)
    return FoldAudit(
        r_sq=cand.r_sq, M=cand.M, shells=cand.shells,
        geo_c6=str(geo_c6), geo_d6=str(geo_d6),
        label_c6_z2=str(lab_c6), label_d6_z2=str(lab_d6),
        geo_c6_f=float(geo_c6), geo_d6_f=float(geo_d6),
        label_c6_z2_f=float(lab_c6), label_d6_z2_f=float(lab_d6),
    )


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Search a generous range of inversion radii and a moderate energy bound; the
    # the canonical radial dual object here is r_sq = 48, max_norm_sq = 240 (M = 42).
    r_sq_values = list(range(1, 61))
    max_norm_sq = 240

    family = enumerate_family(r_sq_values, max_norm_sq)

    print("=" * 78)
    print("RADIAL DUAL FAMILY  (max_norm_sq = %d)" % max_norm_sq)
    print("=" * 78)
    print(f"{'r_sq':>5} {'M':>6} {'#shells':>8} {'uniform':>8}  shells")
    for c in family:
        print(f"{c.r_sq:>5} {c.M:>6} {len(c.shells):>8} "
              f"{str(c.phase_pair_uniform):>8}  {list(c.shells)}")

    print()
    print("=" * 78)
    print("EXACT FOLD FACTORS  (Burnside; reported as exact fractions)")
    print("=" * 78)
    print(f"{'r_sq':>5} {'M':>6} {'geoZ6':>7} {'geoD6':>8} "
          f"{'labZ6xZ2':>10} {'labD6xZ2':>10}")
    audits = []
    for c in family:
        au = audit_candidate(c)
        audits.append(au)
        print(f"{c.r_sq:>5} {c.M:>6} {au.geo_c6:>7} {au.geo_d6:>8} "
              f"{au.label_c6_z2:>10} {au.label_d6_z2:>10}")

    payload = {
        "max_norm_sq": max_norm_sq,
        "r_sq_values": r_sq_values,
        "family": [asdict(c) for c in family],
        "fold_audit": [asdict(a) for a in audits],
    }
    with open("mark4_admissibility.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("\nwrote mark4_admissibility.json")


if __name__ == "__main__":
    main()
