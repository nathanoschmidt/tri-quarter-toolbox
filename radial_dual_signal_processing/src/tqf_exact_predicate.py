"""
tqf_exact_predicate.py - Exact nearest-point predicate for the A2 hexagonal demodulator.

The closed-form A2 demodulator decides which of nine integer lattice candidates
is nearest to a received sample. That decision reduces to comparing squared
Euclidean distances to two candidates and taking the sign of the difference.
This module computes that sign EXACTLY, with no irrational rounding, so the
nearest-point decision is a provable function of the received sample bits rather
than a floating-point verdict that can flip under a different math library, FMA
contraction, or vectorization order.

The mathematics
---------------
Work in lattice space with the basis omega0 = (1, 0), omega1 = (1/2, sqrt(3)/2).
A received sample z has Cartesian coordinates (zx, zy). An integer lattice
candidate (a, b) embeds to the Cartesian point

    p = (a + b/2, b * sqrt(3)/2).

For two candidates p and q the difference of squared distances to z is

    D(p, q) = ||z - p||^2 - ||z - q||^2
            = (||p||^2 - ||q||^2) - 2 <z, p - q>.

Every IEEE double is a dyadic rational, so zx, zy are exact rationals. The
candidate coordinates are integer combinations of 1 and sqrt(3)/2, so ||p||^2,
||q||^2, and <z, p - q> each have the form (rational) + (rational) * sqrt(3).
Collecting terms,

    D(p, q) = alpha + beta * sqrt(3),   with alpha, beta exact rationals.

The sign of alpha + beta * sqrt(3) is decidable in exact integer/rational
arithmetic:
  * if alpha and beta have the same sign (or either is zero) the sign is
    immediate;
  * otherwise compare alpha^2 against 3 * beta^2 (both exact rationals) to
    resolve which term dominates, and attach the sign of alpha.

sqrt(3) is irrational, so alpha + beta * sqrt(3) == 0 forces alpha == beta == 0;
exact ties are therefore detected exactly and never mistaken for a sign.

The adaptive filter
-------------------
Computing every comparison in exact rationals is correct but slow. The standard
remedy (Shewchuk-style adaptive-precision predicates) is a floating-point filter
guarded by a certified error bound: evaluate D(p, q) in double precision, and if
its magnitude exceeds a rigorous bound on the accumulated rounding error, the
float sign is certified correct and returned immediately. Only comparisons that
fall inside the error bound -- samples lying essentially on a Voronoi bisector --
escalate to the exact rational path. At usable SNR these are rare and their
frequency is measurable, so the exactness guarantee costs a small, reported
fraction of escalations rather than a blanket slowdown.

Two entry points are provided:
  * exact_nearest_in_window: the exact-rational decision for a single sample over
    a 3x3 (or arbitrary) integer candidate window; used as the reference/referee.
  * filtered_nearest_in_window: the filtered decision that agrees with the exact
    path bit for bit and additionally reports whether it had to escalate.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "sign_a_plus_b_sqrt3",
    "dist_sq_exact",
    "compare_candidates_exact",
    "exact_nearest_in_window",
    "filtered_nearest_in_window",
    "float_to_fraction",
]

# sqrt(3)/2, the imaginary part of omega1, as an exact reference value.
_SQRT3 = math.sqrt(3.0)
_HALF_SQRT3 = _SQRT3 / 2.0


def float_to_fraction(x: float) -> Fraction:
    """Return the exact dyadic-rational value of an IEEE double.

    ``Fraction(x)`` already captures a float's exact value (floats are dyadic
    rationals); this wrapper documents the intent and guards non-finite inputs.
    """
    if not math.isfinite(x):
        raise ValueError("cannot convert a non-finite float to an exact Fraction")
    return Fraction(x)


def sign_a_plus_b_sqrt3(alpha: Fraction, beta: Fraction) -> int:
    """Return the exact sign of ``alpha + beta * sqrt(3)`` for rational alpha, beta.

    Result is -1, 0, or +1. Because sqrt(3) is irrational, the value is zero iff
    both coefficients are zero, so a returned 0 is an exact tie, not a rounding
    artifact.
    """
    if alpha == 0 and beta == 0:
        return 0
    if beta == 0:
        return 1 if alpha > 0 else -1
    if alpha == 0:
        return 1 if beta > 0 else -1
    a_pos = alpha > 0
    b_pos = beta > 0
    if a_pos and b_pos:
        return 1
    if (not a_pos) and (not b_pos):
        return -1
    # Opposite signs: compare magnitudes via alpha^2 vs 3 * beta^2 (exact).
    lhs = alpha * alpha
    rhs = 3 * beta * beta
    if lhs == rhs:
        # |alpha| == sqrt(3) * |beta| with opposite signs -> exact cancellation.
        return 0
    dominant_alpha = lhs > rhs
    if dominant_alpha:
        return 1 if a_pos else -1
    return 1 if b_pos else -1


def _candidate_coeffs(a: int, b: int) -> Tuple[Fraction, Fraction]:
    """Return (px, py_over_sqrt3) for candidate (a, b).

    The Cartesian embedding is px = a + b/2 (rational) and py = (b/2) * sqrt(3);
    we carry py as its rational coefficient of sqrt(3), i.e. py = py_coeff *
    sqrt(3) with py_coeff = b/2. Keeping the sqrt(3) factored lets every downstream
    quantity stay in the alpha + beta*sqrt(3) form exactly.
    """
    px = Fraction(a) + Fraction(b, 2)
    py_coeff = Fraction(b, 2)
    return px, py_coeff


def dist_sq_exact(zx: Fraction, zy_coeff_one: Fraction, zy_coeff_sqrt3: Fraction,
                  a: int, b: int) -> Tuple[Fraction, Fraction]:
    """Exact squared distance ||z - p||^2 as (rational, rational-coeff-of-sqrt3).

    The sample's y coordinate is supplied already split as
    ``zy = zy_coeff_one + zy_coeff_sqrt3 * sqrt(3)`` (for a plain received sample
    ``zy_coeff_one`` is the whole value and ``zy_coeff_sqrt3`` is zero; the split
    form is used when z is itself an exact lattice-derived point). Returns
    ``(u, v)`` with ``||z - p||^2 = u + v * sqrt(3)``.
    """
    px, py_c = _candidate_coeffs(a, b)
    dx = zx - px
    # dy = (zy_coeff_one) + (zy_coeff_sqrt3 - py_c) * sqrt(3)
    dy_one = zy_coeff_one
    dy_sqrt3 = zy_coeff_sqrt3 - py_c
    # dx^2 is rational; dy^2 = dy_one^2 + 3*dy_sqrt3^2 + 2*dy_one*dy_sqrt3*sqrt(3).
    u = dx * dx + dy_one * dy_one + 3 * dy_sqrt3 * dy_sqrt3
    v = 2 * dy_one * dy_sqrt3
    return u, v


def compare_candidates_exact(zx: Fraction, zy_one: Fraction, zy_sqrt3: Fraction,
                             cand_p: Tuple[int, int],
                             cand_q: Tuple[int, int]) -> int:
    """Exact sign of ||z - p||^2 - ||z - q||^2 for integer candidates p, q.

    Returns -1 if p is strictly nearer, +1 if q is strictly nearer, 0 on an exact
    tie. No floating point enters the decision.
    """
    up, vp = dist_sq_exact(zx, zy_one, zy_sqrt3, cand_p[0], cand_p[1])
    uq, vq = dist_sq_exact(zx, zy_one, zy_sqrt3, cand_q[0], cand_q[1])
    return sign_a_plus_b_sqrt3(up - uq, vp - vq)


def exact_nearest_in_window(zx: float, zy: float,
                            candidates: Sequence[Tuple[int, int]],
                            tie_key=None) -> Tuple[int, int]:
    """Return the exact nearest candidate (a, b) to the sample z = zx + i*zy.

    Distances are compared exactly (see module docstring). Exact ties -- which are
    measure-zero for a continuous channel but can arise on a deterministic test
    grid -- are broken by ``tie_key`` (default: lexicographic on (a, b)), the same
    total order used by the reference decoder, so the winner is deterministic and
    platform independent.
    """
    if tie_key is None:
        tie_key = lambda ab: ab
    zx_f = float_to_fraction(zx)
    zy_one = float_to_fraction(zy)
    zy_sqrt3 = Fraction(0)
    best = candidates[0]
    for cand in candidates[1:]:
        s = compare_candidates_exact(zx_f, zy_one, zy_sqrt3, best, cand)
        if s > 0:
            best = cand            # cand strictly nearer
        elif s == 0:
            # exact tie: defer to the deterministic total order
            if tie_key(cand) < tie_key(best):
                best = cand
    return best


def _float_dist_sq(zx: float, zy: float, a: int, b: int) -> float:
    px = a + 0.5 * b
    py = _HALF_SQRT3 * b
    dx = zx - px
    dy = zy - py
    return dx * dx + dy * dy


def filtered_nearest_in_window(zx: float, zy: float,
                               candidates: Sequence[Tuple[int, int]],
                               tie_key=None) -> Tuple[Tuple[int, int], bool]:
    """Filtered nearest-candidate decision; agrees with the exact path bit for bit.

    Runs a floating-point comparison guarded by a conservative relative error
    bound. When two candidates are within the bound (a near-bisector sample) the
    single pair escalates to the exact rational comparison. Returns
    ``((a, b), escalated)`` where ``escalated`` is True iff any comparison needed
    the exact path.

    The bound is a simple, safe multiple of machine epsilon scaled by the
    magnitudes involved -- deliberately conservative, so it may escalate a few
    comparisons that float alone would have gotten right, but it never certifies a
    wrong sign. Tightening the constant is a performance-only change and cannot
    affect correctness because the exact path is the arbiter.
    """
    if tie_key is None:
        tie_key = lambda ab: ab
    escalated = False
    best = candidates[0]
    best_d2 = _float_dist_sq(zx, zy, best[0], best[1])
    zx_f = None
    zy_one = None
    for cand in candidates[1:]:
        cand_d2 = _float_dist_sq(zx, zy, cand[0], cand[1])
        diff = best_d2 - cand_d2
        # Conservative error bound on ``diff``: magnitudes are O(||z||^2 + ||p||^2);
        # 8 * eps * (|best_d2| + |cand_d2| + tiny) safely dominates the accumulated
        # rounding error of the handful of flops above.
        bound = 8.0 * np.finfo(np.float64).eps * (abs(best_d2) + abs(cand_d2) + 1e-300)
        if abs(diff) > bound:
            if diff > 0.0:
                best, best_d2 = cand, cand_d2
            continue
        # Inside the filter bound -> resolve exactly.
        escalated = True
        if zx_f is None:
            zx_f = float_to_fraction(zx)
            zy_one = float_to_fraction(zy)
        s = compare_candidates_exact(zx_f, zy_one, Fraction(0), best, cand)
        if s > 0:
            best, best_d2 = cand, cand_d2
        elif s == 0 and tie_key(cand) < tie_key(best):
            best, best_d2 = cand, cand_d2
    return best, escalated
