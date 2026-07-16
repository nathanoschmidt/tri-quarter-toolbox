"""
test_tqf_admissibility.py - Radial Dual Family Enumeration and Exact Fold-Factor Audit Tests

Exact, integer-only tests for the two structural questions the design-search and
fold-at-scale studies depend on:

  * which radial dual constellations exist (shell sets closed under circle
    inversion N -> r^4 / N), and their orders M; and
  * by how much symmetry folds the work -- the Burnside fold factors for the
    geometric groups Z6 / D6 and the label groups Z6 x Z2 / D6 x Z2, each equal
    to |points| / |orbits| exactly.

Everything is verified with exact integer / Fraction arithmetic (no tolerance):
shells are integer Eisenstein norms, group actions are integer coordinate maps,
and inversion acts on integer shell labels.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 6, 2026
"""

from fractions import Fraction

import pytest

import tqf_admissibility as adm


# --------------------------------------------------------------------------- #
# Exact lattice primitives
# --------------------------------------------------------------------------- #
def test_shell_norm_sq_is_eisenstein_norm():
    assert adm.shell_norm_sq(1, 0) == 1
    assert adm.shell_norm_sq(1, 1) == 3
    assert adm.shell_norm_sq(2, 1) == 7
    assert adm.shell_norm_sq(-2, 1) == 3          # rotation/reflection invariant


def test_rotate60_has_order_six():
    for (a, b) in [(1, 0), (2, 1), (-3, 2)]:
        p = (a, b)
        for _ in range(6):
            p = adm.rotate60(*p)
        assert p == (a, b)                        # R applied six times is identity


def test_reflect_is_an_involution():
    for (a, b) in [(1, 0), (2, 1), (-3, 2)]:
        assert adm.reflect(*adm.reflect(a, b)) == (a, b)


def test_loeschian_shells_include_norms_exclude_nonnorms():
    shells = set(adm.loeschian_shells_up_to(20))
    assert {1, 3, 4, 7, 9, 12, 13, 16, 19} <= shells
    assert 2 not in shells and 5 not in shells and 6 not in shells


@pytest.mark.parametrize("n", [1, 3, 4, 7, 9, 12, 48])
def test_shell_size_is_a_multiple_of_six(n):
    assert adm.shell_size(n) % 6 == 0


# --------------------------------------------------------------------------- #
# Radial dual family enumeration (closed under N -> r^4 / N)
# --------------------------------------------------------------------------- #
def test_shell_pairs_for_r_sq_12_are_the_four_integer_duals():
    pairs = adm.radial_dual_shell_pairs(12, 48)
    assert pairs == [(3, 48), (4, 36), (9, 16), (12, 12)]
    # every pair multiplies to r^4 = 144^1 ... i.e. r_sq^2 = 144
    for (n, nd) in pairs:
        assert n * nd == 12 * 12 * 1 * 1 or n * nd == 144


def test_build_candidate_canonical_m42_shape():
    c = adm.build_candidate(12, 48)
    assert c.M == 42
    assert c.shells == (3, 4, 9, 12, 16, 36, 48)
    assert c.self_dual_shell == 12
    assert c.phase_pair_uniform is True
    assert c.k_per_sector == (1, 1, 1, 1, 1, 1, 1)   # one point per sector per shell


def test_build_candidate_returns_none_without_dual_pair():
    # A radius whose r^4 admits no in-bound integer-dual shell pair -> not admissible.
    assert adm.build_candidate(5, 6) is None


def test_enumerate_family_is_a_deduplicated_size_ladder():
    fam = adm.enumerate_family(range(1, 60), 84)
    sizes = [c.M for c in fam]
    assert sizes == sorted(sizes)                 # sorted by order M
    shell_sets = [c.shells for c in fam]
    assert len(shell_sets) == len(set(shell_sets))  # deduplicated by shell set


# --------------------------------------------------------------------------- #
# Exact Burnside fold factors (|points| / |orbits|)
# --------------------------------------------------------------------------- #
def _canonical_points():
    c = adm.build_candidate(12, 48)
    return [p for n in c.shells for p in adm.shell_points(n)], c


def test_geometric_fold_is_six_on_the_canonical_object():
    pts, _ = _canonical_points()
    # Origin-free, Z6-closed: every orbit has size 6, so the fold is exactly 6.
    assert adm.burnside_geometric_fold(pts, include_reflections=False) == Fraction(6)
    assert adm.burnside_geometric_fold(pts, include_reflections=True) == Fraction(6)


def test_geometric_fold_equals_points_over_orbits():
    # Cross-check Burnside against a direct orbit count on a single shell.
    pts = adm.shell_points(7)
    orbits = set()
    seen = set()
    elems = adm._group_elements_geometric(include_reflections=False)
    for p in pts:
        if p in seen:
            continue
        orbit = {act(*p) for act in elems}
        orbits.add(frozenset(orbit))
        seen |= orbit
    direct = Fraction(len(pts), len(orbits))
    assert adm.burnside_geometric_fold(pts, include_reflections=False) == direct


def test_label_fold_c6_z2_is_ten_and_a_half_at_m42():
    _, c = _canonical_points()
    fold = adm.burnside_label_fold(list(c.shells), 12, include_reflections=False)
    assert fold == Fraction(21, 2)                # 10.5x label fold at M = 42


def test_burnside_rejects_non_group_closed_set():
    # A partial shell (not closed under the rotation) has a non-integral orbit
    # count, which the exact fold routine must reject rather than round.
    partial = adm.shell_points(7)[:3]
    with pytest.raises(ValueError):
        adm.burnside_geometric_fold(partial, include_reflections=False)
