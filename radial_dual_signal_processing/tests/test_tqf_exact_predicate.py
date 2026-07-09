"""
test_tqf_exact_predicate.py - Correctness tests for the exact nearest-point predicate.

Verifies the exact Z[sqrt(3)] sign routine, agreement between the exact and
float-filtered decision paths, and agreement of the exact decision with a
brute-force float argmin except at genuine near-bisector ties (where the exact
path is authoritative by construction).

Run: pytest -q test_tqf_exact_predicate.py

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 8, 2026
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

import tqf_exact_predicate as ep


H = math.sqrt(3.0) / 2.0


def _emb(a: int, b: int):
    return (a + 0.5 * b, H * b)


def test_sign_basic():
    assert ep.sign_a_plus_b_sqrt3(Fraction(0), Fraction(0)) == 0
    assert ep.sign_a_plus_b_sqrt3(Fraction(1), Fraction(0)) == 1
    assert ep.sign_a_plus_b_sqrt3(Fraction(-1), Fraction(0)) == -1
    assert ep.sign_a_plus_b_sqrt3(Fraction(0), Fraction(3)) == 1
    assert ep.sign_a_plus_b_sqrt3(Fraction(0), Fraction(-3)) == -1


def test_sign_opposite_terms():
    # 2 - sqrt(3) ~ +0.268
    assert ep.sign_a_plus_b_sqrt3(Fraction(2), Fraction(-1)) == 1
    # 1 - sqrt(3) ~ -0.732
    assert ep.sign_a_plus_b_sqrt3(Fraction(1), Fraction(-1)) == -1
    # -2 + sqrt(3) ~ -0.268
    assert ep.sign_a_plus_b_sqrt3(Fraction(-2), Fraction(1)) == -1


def test_sign_exact_cancellation():
    # alpha^2 == 3 beta^2 with opposite signs -> exact zero.
    # Take alpha = 3, beta = -sqrt(3) is irrational; instead use rationals with
    # alpha^2 = 3 beta^2: alpha = 3, beta = -Fraction(3) gives 9 vs 27 (no).
    # A true rational cancellation needs alpha/beta = sqrt(3), impossible for
    # nonzero rationals, so the only exact zero is alpha == beta == 0 -- already
    # covered. This test documents that no spurious zero is returned.
    assert ep.sign_a_plus_b_sqrt3(Fraction(3), Fraction(-1)) == 1   # 9 > 3
    assert ep.sign_a_plus_b_sqrt3(Fraction(1), Fraction(-2)) == -1  # 1 < 12


def test_exact_matches_float_argmin():
    rng = np.random.default_rng(0)
    real_mismatches = 0
    for _ in range(100000):
        zx, zy = rng.uniform(-3, 3), rng.uniform(-3, 3)
        b0 = zy / H
        a0 = zx - 0.5 * b0
        na, nb = round(a0), round(b0)
        cands = [(na + da, nb + db) for da in (-1, 0, 1) for db in (-1, 0, 1)]
        d2 = [(zx - _emb(*c)[0]) ** 2 + (zy - _emb(*c)[1]) ** 2 for c in cands]
        bf = cands[int(np.argmin(d2))]
        ex = ep.exact_nearest_in_window(zx, zy, cands)
        if ex != bf:
            s = ep.compare_candidates_exact(Fraction(zx), Fraction(zy),
                                            Fraction(0), ex, bf)
            if s > 0:  # exact says bf strictly nearer but we returned ex
                real_mismatches += 1
    assert real_mismatches == 0


def test_filtered_equals_exact():
    rng = np.random.default_rng(1)
    for _ in range(100000):
        zx, zy = rng.uniform(-3, 3), rng.uniform(-3, 3)
        b0 = zy / H
        a0 = zx - 0.5 * b0
        na, nb = round(a0), round(b0)
        cands = [(na + da, nb + db) for da in (-1, 0, 1) for db in (-1, 0, 1)]
        ex = ep.exact_nearest_in_window(zx, zy, cands)
        fl, _ = ep.filtered_nearest_in_window(zx, zy, cands)
        assert fl == ex


def test_engineered_bisector_escalates():
    # A sample exactly on the perpendicular bisector of (0,0) and (1,0) in
    # lattice space has zx = 0.5, zy on the axis; the filter should escalate and
    # the exact tie-break should pick the lexicographically smaller candidate.
    zx, zy = 0.5, 0.0
    cands = [(0, 0), (1, 0)]
    ex = ep.exact_nearest_in_window(zx, zy, cands)
    fl, esc = ep.filtered_nearest_in_window(zx, zy, cands)
    assert ex == fl == (0, 0)   # tie broken by lexicographic order
    assert esc is True
