"""
tqf_hex_signal.py - Core Hexagonal Signal-Processing Library for the TQF Radial Dual Subproject

The shared, exact signal-processing library for the Tri-Quarter Framework (TQF)
radial_dual_signal_processing subproject.

This module provides the exact, reusable building blocks for the experiments:

  * The base triangular (Eisenstein/A2) lattice L with basis omega0 = 1 and
    omega1 = exp(i*pi/3), and the order-6 rotation R(a, b) = (-b, a + b).
  * Hexagonal signal constellations carved from L:
        - build_filled_constellation(M): the M lowest-energy lattice points
          (textbook minimum-energy "hexagonal QAM"); used for the binary
          power-of-two comparisons against square QAM.
        - build_disk_constellation(max_norm_sq): all non-origin lattice points
          within a squared-norm threshold -- exactly 6-fold (Z6) symmetric --
          used for the symmetry-reduced exact-metric study.
  * An exact, constant-time (O(1) per symbol) closed-form lattice demodulator
    (a 3x3 candidate-window A2 nearest-point search) with a constellation-
    membership test and an exhaustive fallback for the rare exterior symbol,
    so that its decisions are bitwise-identical to exhaustive maximum-likelihood
    (ML) nearest-point decoding -- by construction, and verified empirically.
  * Square M-QAM (Gray-mapped, unit energy) as an apples-to-apples baseline.
  * Exact integer/rational primitives: the sector index (integer cross-product
    sign test), the shell index (integer Eisenstein norm a^2 + a*b + b^2), the
    color residue, and the circle inversion iota_r (exact rational, involutive).
  * Channels generalizing the BPSK case study to 2D: complex AWGN, 2D
    impulsive noise, and flat Rayleigh fading (with per-symbol gains for
    perfect-CSI / zero-forcing reception).
  * Differential hexagonal (senary) sector coding for the rotation-robustness
    study, plus the exact Clopper-Pearson binomial confidence interval.

Design notes / conventions
---------------------------
* "Lattice space" uses integer oblique coordinates (a, b) with the complex
  embedding a*omega0 + b*omega1. "Signal space" uses unit-average-energy complex
  symbols s * (lattice point), where s is the energy-normalization scale.
* Where the framework claims exactness, the computation is integer or
  fractions.Fraction (no floating point): sector, shell, color, inversion, and
  the symmetry-reduced metric. Floating point is confined to the channel
  (noise + the Euclidean metric used identically by both decoders, so it
  cancels in the comparison).
* All randomness is drawn from an explicitly supplied numpy.random.Generator so
  every experiment is deterministic and reproducible under a documented seed.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.1.0
Date: June 27, 2026
"""

from __future__ import annotations

import datetime
import json
import math
import os
import platform
import socket
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Base lattice constants (Eisenstein / A2)
# ---------------------------------------------------------------------------

# Real/imaginary parts of the two basis vectors omega0 = 1, omega1 = exp(i*pi/3).
_OMEGA1_RE = 0.5
_OMEGA1_IM = math.sqrt(3.0) / 2.0
OMEGA0 = complex(1.0, 0.0)
OMEGA1 = complex(_OMEGA1_RE, _OMEGA1_IM)

# The six primary ray directions d_k = R^k(1, 0), as integer (a, b) pairs.
# R is the +60 degree rotation R(a, b) = (-b, a + b); see rotate60().
PRIMARY_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),    # 0   degrees  (omega0)
    (0, 1),    # 60  degrees  (omega1)
    (-1, 1),   # 120 degrees
    (-1, 0),   # 180 degrees
    (0, -1),   # 240 degrees
    (1, -1),   # 300 degrees
)


def rotate60(a: int, b: int) -> Tuple[int, int]:
    """Return R(a, b) = (-b, a + b), the exact +60 degree lattice rotation.

    R is an order-6 integer automorphism of the triangular lattice
    (R**6 == identity) realizing the Eisenstein unit-group action.
    """
    return (-b, a + b)


def lattice_to_complex(a: int, b: int) -> complex:
    """Map integer oblique coordinates (a, b) to the complex point a*omega0 + b*omega1."""
    return complex(a + b * _OMEGA1_RE, b * _OMEGA1_IM)


def lattice_array_to_complex(ab: np.ndarray) -> np.ndarray:
    """Vectorized lattice_to_complex for an (N, 2) integer array of (a, b) rows."""
    a = ab[:, 0].astype(np.float64)
    b = ab[:, 1].astype(np.float64)
    return (a + b * _OMEGA1_RE) + 1j * (b * _OMEGA1_IM)


def complex_to_oblique(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert the basis: recover real oblique coordinates (a, b) from complex z.

    From z = a + b*omega1 we have Re(z) = a + b/2 and Im(z) = b*sqrt(3)/2, hence
    b = Im(z) / (sqrt(3)/2) and a = Re(z) - b/2. These are real (not yet rounded).
    """
    z = np.asarray(z)
    b = np.imag(z) / _OMEGA1_IM
    a = np.real(z) - b * _OMEGA1_RE
    return a, b


# ---------------------------------------------------------------------------
# Exact integer/rational primitives: shell, sector, color, inversion
# ---------------------------------------------------------------------------

def shell_norm_sq(a: int, b: int) -> int:
    """Return the integer Eisenstein norm a^2 + a*b + b^2 = squared Euclidean length.

    This is the exact squared distance of (a, b) from the origin in signal space
    (up to the global energy-normalization scale) and the radial "shell" key.
    """
    return a * a + a * b + b * b


def _det(ua: int, ub: int, va: int, vb: int) -> int:
    """Integer determinant ua*vb - va*ub; its sign equals the sign of the planar
    cross product of the embeddings of (ua, ub) and (va, vb)."""
    return ua * vb - va * ub


def phase_pair_sector(a: int, b: int) -> int:
    """Return the exact phase-pair sector index in {0, ..., 5} of a point (a, b).

    *The phase-pair primitive.* This is the framework's central exact, floating-
    point-free angular coordinate: it locates a non-origin lattice point in one
    of the six 60 degree wedges spanned by the primary ray pairs
    (d_k, d_{k+1}) by testing the sign of two integer cross products (the
    "phase pair") -- ``det(d_k, p) >= 0`` AND ``det(p, d_{k+1}) > 0``. Sector S_k
    is the half-open wedge [d_k, d_{k+1}); a point lying exactly on the primary
    ray d_k is assigned to sector k. Because every test is an integer
    determinant, the sector is exact (no floating-point phase ever enters), which
    is what makes labelling, orbit partitioning, equivariance, and the
    differential codec lossless and integer-only. The origin has no defined
    sector and returns -1.

    This primitive is used *everywhere* a sector is needed (constellation build,
    folded decoder, metric orbit reduction, differential coding); ``sector_index``
    is retained as a backward-compatible alias.
    """
    if a == 0 and b == 0:
        return -1
    for k in range(6):
        da, db = PRIMARY_DIRECTIONS[k]
        ea, eb = PRIMARY_DIRECTIONS[(k + 1) % 6]
        # On ray d_k, or strictly CCW of d_k, AND strictly CW of d_{k+1}.
        if _det(da, db, a, b) >= 0 and _det(a, b, ea, eb) > 0:
            return k
    # Fallback (should not occur); assign by closest primary ray.
    return -1


def phase_pair_sector_array(ab: np.ndarray) -> np.ndarray:
    """Vectorized exact phase-pair sector for an (N, 2) integer array; origin -> -1.

    The batched form of :func:`phase_pair_sector`; identical integer-only
    cross-product logic applied to every row at once.
    """
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    out = np.full(a.shape, -1, dtype=np.int64)
    assigned = np.zeros(a.shape, dtype=bool)
    for k in range(6):
        da, db = PRIMARY_DIRECTIONS[k]
        ea, eb = PRIMARY_DIRECTIONS[(k + 1) % 6]
        det_k = da * b - a * db          # _det(d_k, p)
        det_kp1 = a * eb - ea * b        # _det(p, d_{k+1})
        hit = (~assigned) & (det_k >= 0) & (det_kp1 > 0)
        out[hit] = k
        assigned |= hit
    out[(a == 0) & (b == 0)] = -1
    return out


# Backward-compatible aliases: the sector index *is* the phase-pair sector. The
# ``phase_pair_sector`` name is the documented first-class primitive (Mark 2);
# ``sector_index`` / ``sector_index_array`` are kept so existing callers and the
# earlier studies/tests continue to work unchanged.
sector_index = phase_pair_sector
sector_index_array = phase_pair_sector_array


def _phase_pair_sector_rational(a: Fraction, b: Fraction) -> int:
    """Exact phase-pair sector for *rational* oblique coordinates (a, b).

    Identical sign-of-cross-product logic as :func:`phase_pair_sector`, but with
    Fraction inputs, so the sector of an exact (non-lattice) point such as a
    circle-inversion image can be tested without any floating-point phase. Used
    to verify the commutativity lemma sector(iota_r(v)) == sector(v) exactly.
    """
    if a == 0 and b == 0:
        return -1
    for k in range(6):
        da, db = PRIMARY_DIRECTIONS[k]
        ea, eb = PRIMARY_DIRECTIONS[(k + 1) % 6]
        det_k = Fraction(da) * b - a * Fraction(db)       # det(d_k, p)
        det_kp1 = a * Fraction(eb) - Fraction(ea) * b     # det(p, d_{k+1})
        if det_k >= 0 and det_kp1 > 0:
            return k
    return -1


def color_residue(a: int, b: int) -> int:
    """Return the exact 3-coloring residue c3 = (a - b) mod 3.

    For the triangular lattice this is a proper 3-coloring (adjacent vertices
    never share c3); see tqf_lattice_graph.six_coloring for the equivariant
    refinement to the trihexagonal six-coloring used by the parallel benchmark.
    """
    return (a - b) % 3


def inversion(a: int, b: int, r_sq: int) -> complex:
    """Exact circle inversion iota_r of lattice point (a, b) about radius r (r^2 = r_sq).

    iota_r(v) = (r^2 / ||v||^2) * v, computed in exact rational arithmetic and
    returned as a complex number with rational real/imaginary parts cast to float
    only at the end. iota_r is an involution (iota_r(iota_r(v)) == v) and fixes
    the boundary ||v||^2 == r_sq; it swaps the inner and outer zones while
    preserving the angular sector. The image is generally not a lattice point.
    """
    n_sq = shell_norm_sq(a, b)
    if n_sq == 0:
        raise ValueError("inversion is undefined at the origin (punctured)")
    scale = Fraction(r_sq, n_sq)
    # Embedding: x = a + b/2, y = b*sqrt(3)/2. Keep the rational part exact; the
    # irrational sqrt(3)/2 factor is common and applied at the end.
    x_rat = scale * (Fraction(a) + Fraction(b, 2))
    y_rat_over_sqrt = scale * Fraction(b)  # multiply by sqrt(3)/2 below
    return complex(float(x_rat), float(y_rat_over_sqrt) * _OMEGA1_IM)


def inversion_exact(a: int, b: int, r_sq: int) -> Tuple[Fraction, Fraction]:
    """Return iota_r(v) as exact oblique rational coordinates (a', b').

    Because R(a, b) keeps the basis, the inverse map is linear in (a, b); here we
    return the oblique coordinates as exact Fractions so that the involution can
    be checked with ``==`` rather than a floating-point tolerance.
    """
    n_sq = shell_norm_sq(a, b)
    if n_sq == 0:
        raise ValueError("inversion is undefined at the origin (punctured)")
    scale = Fraction(r_sq, n_sq)
    return (scale * Fraction(a), scale * Fraction(b))


# ---------------------------------------------------------------------------
# Exact inversion duality on LABELS (the Z2 involution used for storage folding)
# ---------------------------------------------------------------------------
#
# Circle inversion iota_r is conformal, NOT isometric, so it never enters a
# Euclidean distance or an ML decision (the "inversion firewall"). What it does
# give -- exactly, in integer/rational arithmetic -- is an involutive duality on
# the *labels* (phase-pair sector, shell). It preserves the sector (the phase-
# pair test commutes with iota_r; this is the commutativity lemma, established as
# Proposition 4.15 of the lattice-graph paper [1] -- cited here, not re-derived,
# and verified empirically by verify_inversion_commutativity below) and maps the
# squared-norm shell N to the dual shell r^4 / N. When r^4 is divisible by N and
# the quotient is itself an Eisenstein norm, the dual shell is an exact integer
# shell, so inner and outer complete shells pair up with identical sector
# occupancy. This is what lets storage, precomputation, and label tables fold to
# the fundamental (inner) domain while every Euclidean decision stays exact.

def invert_sector_shell(s6: int, n_sq: int, r_sq: int) -> Tuple[int, Fraction]:
    """Exact label-space inversion: (sector, shell N) -> (sector, dual shell r^4/N).

    The sector ``s6`` is returned unchanged (iota_r preserves the phase-pair
    sector exactly), and the shell ``n_sq`` maps to the exact rational
    ``Fraction(r_sq**2, n_sq)``. When that Fraction is an integer it is a genuine
    dual shell norm; when it equals ``n_sq`` the shell lies on the inversion
    circle (self-dual boundary). Raises on the punctured origin (no shell).
    """
    if n_sq <= 0:
        raise ValueError("shell inversion is undefined at the origin (N <= 0)")
    return s6, Fraction(r_sq * r_sq, n_sq)


def dual_shell_norm(n_sq: int, r_sq: int) -> int | None:
    """Return the integer dual shell r^4 / N if it is an exact integer, else None.

    Convenience wrapper over :func:`invert_sector_shell` for the common case of
    folding a complete-shell membership table: an outer shell ``n_sq`` is a
    member of an inversion-paired constellation iff its integer dual lies in the
    stored inner-shell set.
    """
    r4 = r_sq * r_sq
    if n_sq > 0 and r4 % n_sq == 0:
        return r4 // n_sq
    return None


def invert_label(a: int, b: int, r_sq: int) -> Tuple[Fraction, Fraction, int, Fraction]:
    """Full exact inversion of a lattice label about radius r (r^2 = r_sq).

    Returns ``(a', b', sector, dual_shell)`` where ``(a', b')`` are the exact
    rational oblique coordinates of iota_r(v) (generally not a lattice point),
    ``sector`` is the phase-pair sector (identical for v and iota_r(v) by the
    commutativity lemma), and ``dual_shell`` is the exact rational dual shell
    ``r^4 / ||v||^2``. This is the label-space Z2 involution: applying it twice
    returns the original (sector, shell) and the original rational coordinates.
    """
    af, bf = inversion_exact(a, b, r_sq)
    s6 = phase_pair_sector(a, b)
    _, dual = invert_sector_shell(s6, shell_norm_sq(a, b), r_sq)
    return af, bf, s6, dual


def verify_inversion_commutativity(r_sq: int, coord_radius: int = 8
                                   ) -> Tuple[int, int, int]:
    """Empirically certify the two exact facts that make inversion folding valid.

    Over every non-origin lattice point with |a|, |b| <= ``coord_radius``:
      * the involution iota_r(iota_r(v)) == v holds exactly (rational ==), and
      * the commutativity lemma sector(iota_r(v)) == sector(v) holds exactly,
        using the rational phase-pair test (no floating point).

    Returns ``(num_checked, involution_violations, commutativity_violations)``;
    both violation counts are expected to be zero. (The lemma itself is
    Proposition 4.15 of the lattice-graph paper [1]; this routine verifies it
    rather than re-deriving it.)
    """
    checked = inv_viol = comm_viol = 0
    for a in range(-coord_radius, coord_radius + 1):
        for b in range(-coord_radius, coord_radius + 1):
            if a == 0 and b == 0:
                continue
            af, bf = inversion_exact(a, b, r_sq)
            # involution: invert the rational image again, must return (a, b).
            n_img = af * af + af * bf + bf * bf
            scale2 = Fraction(r_sq) / n_img
            if (scale2 * af, scale2 * bf) != (Fraction(a), Fraction(b)):
                inv_viol += 1
            if _phase_pair_sector_rational(af, bf) != phase_pair_sector(a, b):
                comm_viol += 1
            checked += 1
    return checked, inv_viol, comm_viol


def orbit_under_rotation(a: int, b: int) -> List[Tuple[int, int]]:
    """Return the (up to 6) distinct images of (a, b) under R^0..R^5."""
    pts: List[Tuple[int, int]] = []
    ca, cb = a, b
    for _ in range(6):
        pts.append((ca, cb))
        ca, cb = rotate60(ca, cb)
    # Deduplicate while preserving order (handles short orbits, e.g., the origin).
    seen: Dict[Tuple[int, int], None] = {}
    for p in pts:
        seen.setdefault(p, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Constellation containers and builders
# ---------------------------------------------------------------------------

@dataclass
class Constellation:
    """A finite signal constellation carved from the lattice (or square QAM).

    Attributes
    ----------
    ab:
        Integer (M, 2) array of oblique lattice coordinates (empty/ignored for
        square QAM, which is not a sublattice of L).
    points_unit:
        Complex (M,) array of unit-average-energy signal points.
    labels:
        Integer (M,) array of bit labels (Gray for square QAM; a documented
        spatial Gray-like labeling for hexagonal constellations).
    bits_per_symbol:
        log2(M).
    scale:
        Float energy-normalization factor s with points_unit = s * (raw points).
    scale_sq_exact:
        Exact rational s^2 (available for lattice constellations).
    name:
        Human-readable constellation name.
    """

    points_unit: np.ndarray
    labels: np.ndarray
    bits_per_symbol: int
    scale: float
    name: str
    ab: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    scale_sq_exact: Fraction | None = None
    # Optional radial-dual (Mark 2) metadata, set only by
    # build_radial_dual_constellation; None/empty for the other constellations.
    shell_norms: Tuple[int, ...] | None = None        # the complete shells present
    inversion_r_sq: int | None = None                 # inversion radius^2 (e.g. 12)
    phase_pair_uniform: bool | None = None            # equal sector occupancy verified
    inversion_paired: bool | None = None              # closed under iota_r duality
    fundamental_domain_size: int | None = None        # |sector 0 inner+boundary|
    inversion_dual_index: np.ndarray | None = None    # (M,) point->inversion-dual point

    @property
    def size(self) -> int:
        return int(self.points_unit.shape[0])


def _gray_sequence_labels(order: Sequence[int], nbits: int) -> np.ndarray:
    """Assign binary-reflected Gray codes to points in the given visiting order.

    ``order`` is a permutation of point indices; consecutive points in the order
    receive Gray-adjacent (1-bit-apart) codewords. When ``order`` lists spatial
    neighbours consecutively, most adjacent constellation points differ by a
    single bit -- a documented, reproducible "Gray-like" labeling. Not claimed
    to be an optimal hexagonal Gray map; SER (label-independent) is the headline
    metric for the packing-gain comparison.
    """
    m = len(order)
    labels = np.empty(m, dtype=np.int64)
    for rank, idx in enumerate(order):
        labels[idx] = rank ^ (rank >> 1)  # standard binary-reflected Gray code
    return labels


def _enumerate_lattice_points(max_radius: int) -> List[Tuple[int, int, int]]:
    """Return (norm_sq, a, b) for all lattice points with |a|,|b| <= max_radius."""
    pts: List[Tuple[int, int, int]] = []
    for a in range(-max_radius, max_radius + 1):
        for b in range(-max_radius, max_radius + 1):
            pts.append((shell_norm_sq(a, b), a, b))
    return pts


def build_filled_constellation(m: int) -> Constellation:
    """Build the minimum-energy hexagonal constellation of size m (a power of two).

    Selects the m lowest-energy lattice points (ties broken deterministically by
    angle then coordinate), normalizes to unit average energy, and assigns a
    spatial Gray-like labeling. This is the standard filled "hexagonal QAM"
    region and the headline object for the binary hex-vs-square comparison.

    Note: m need not be a multiple of 6; the 6k-per-shell structure constrains
    *shells*, not the constellation order. The zone/sector machinery still
    applies for decoding -- it is the decode apparatus, not a size constraint.
    """
    if m < 2 or (m & (m - 1)) != 0:
        raise ValueError("m must be a power of two >= 2 for the binary comparison")
    nbits = int(round(math.log2(m)))
    # A generous radius guarantees at least m points to choose from.
    radius = max(4, int(math.ceil(math.sqrt(m))) + 3)
    cand = _enumerate_lattice_points(radius)

    def sort_key(t: Tuple[int, int, int]):
        n_sq, a, b = t
        ang = math.atan2(b * _OMEGA1_IM, a + b * _OMEGA1_RE) if (a or b) else -math.inf
        return (n_sq, ang, a, b)

    cand.sort(key=sort_key)
    chosen = cand[:m]
    ab = np.array([[a, b] for (_n, a, b) in chosen], dtype=np.int64)
    raw = lattice_array_to_complex(ab)

    sum_norm_sq = int(sum(n for (n, _a, _b) in chosen))
    scale_sq = Fraction(m, sum_norm_sq) if sum_norm_sq > 0 else Fraction(1, 1)
    scale = math.sqrt(float(scale_sq))
    points_unit = raw * scale

    # Spatial Gray-like order: walk points by sector, then radius, then angle.
    def label_order_key(i: int):
        a, b = int(ab[i, 0]), int(ab[i, 1])
        s = sector_index(a, b)
        return (s if s >= 0 else -1, shell_norm_sq(a, b),
                math.atan2(b * _OMEGA1_IM, a + b * _OMEGA1_RE) if (a or b) else 0.0)

    order = sorted(range(m), key=label_order_key)
    labels = _gray_sequence_labels(order, nbits)
    return Constellation(points_unit=points_unit, labels=labels,
                         bits_per_symbol=nbits, scale=scale,
                         name=f"hex-{m}", ab=ab, scale_sq_exact=scale_sq)


def build_disk_constellation(max_norm_sq: int) -> Constellation:
    """Build the 6-fold-symmetric constellation of all non-origin lattice points
    with 0 < ||v||^2 <= max_norm_sq.

    The origin is excluded (consistent with the TQF puncture), so the point set
    is closed under the order-6 rotation R and partitions into full 6-element
    orbits -- the symmetric "core" exploited by the symmetry-reduced metric
    study. Energy-normalized to unit average energy; no bit labeling is attached
    (this constellation is used for exact metric computation, not BER).
    """
    radius = int(math.ceil(math.sqrt(max_norm_sq))) + 2
    chosen = [(n, a, b) for (n, a, b) in _enumerate_lattice_points(radius)
              if 0 < n <= max_norm_sq]
    chosen.sort()
    ab = np.array([[a, b] for (_n, a, b) in chosen], dtype=np.int64)
    raw = lattice_array_to_complex(ab)
    sum_norm_sq = int(sum(n for (n, _a, _b) in chosen))
    scale_sq = Fraction(len(chosen), sum_norm_sq)
    scale = math.sqrt(float(scale_sq))
    points_unit = raw * scale
    labels = np.arange(len(chosen), dtype=np.int64)  # placeholder; unused for BER
    return Constellation(points_unit=points_unit, labels=labels,
                         bits_per_symbol=0, scale=scale,
                         name=f"disk-Nle{max_norm_sq}", ab=ab,
                         scale_sq_exact=scale_sq)


def loeschian_shells_up_to(max_norm_sq: int) -> List[int]:
    """Return the sorted list of positive Eisenstein norms (Loeschian numbers)
    that are realized by lattice points with squared norm <= ``max_norm_sq``.

    A Loeschian number is an integer of the form a^2 + a*b + b^2; these are
    exactly the squared lengths (shell radii^2) of triangular-lattice points.
    """
    radius = int(math.ceil(math.sqrt(max_norm_sq))) + 2
    norms = set()
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            n = shell_norm_sq(a, b)
            if 0 < n <= max_norm_sq:
                norms.add(n)
    return sorted(norms)


def radial_dual_shell_pairs(r_sq: int, max_norm_sq: int
                            ) -> List[Tuple[int, int]]:
    """Return the inversion-dual complete-shell pairs about radius r (r^2 = r_sq).

    A shell ``N`` pairs with ``r^4 / N`` under circle inversion. This returns the
    list of ``(N, N_dual)`` with ``N <= N_dual``, both Eisenstein norms and both
    ``<= max_norm_sq``, that the inversion maps onto each other exactly in integer
    arithmetic. The self-dual shell ``N == r_sq`` (the points lying on the
    inversion circle) appears as ``(r_sq, r_sq)``. With ``r_sq = 12`` and
    ``max_norm_sq = 60`` these are (3, 48), (4, 36), (9, 16), and the self-dual
    (12, 12) -- four equal-occupancy integer-dual shell pairs.
    """
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


def build_radial_dual_constellation(r_sq: int = 12, max_norm_sq: int = 60
                                    ) -> Constellation:
    """Build the phase-pair-uniform, inversion-paired radial-dual constellation (C7).

    This is the "Option B" (shell-complete, full rotation x inversion symmetry)
    constellation: the union of *complete* lattice shells that close under both
    the order-6 rotation R (phase-pair / C6 symmetry) and circle inversion
    iota_r about radius r (radial Z2 duality). Concretely it takes every shell
    that participates in a :func:`radial_dual_shell_pairs` pair about ``r_sq``
    (inner shell, its exact integer dual outer shell, and the self-dual boundary
    shell on the inversion circle), giving genuinely uniform sector occupancy and
    exact inner/outer inversion pairing on a single object. Because ``6`` does not
    divide ``2^m``, a shell-complete constellation cannot also be a power-of-two
    "filled" region, so this object is a symmetry/duality demonstrator (like the
    disk constellation), not a binary-M comparison constellation.

    The returned :class:`Constellation` carries the Mark 2 metadata:
    ``shell_norms``, ``inversion_r_sq``, ``phase_pair_uniform`` (every complete
    shell has exactly ``k`` points per sector, here k=1), ``inversion_paired``
    (closed under the exact iota_r point permutation), ``fundamental_domain_size``
    (one sector restricted to the inner + boundary shells), and
    ``inversion_dual_index`` (each point's exact inversion-dual point index).
    Energy-normalized to unit average energy; no bit labeling is attached.
    """
    pairs = radial_dual_shell_pairs(r_sq, max_norm_sq)
    if not pairs:
        raise ValueError(f"no inversion-dual shell pairs for r_sq={r_sq} "
                         f"within max_norm_sq={max_norm_sq}")
    shell_norms = sorted({n for pair in pairs for n in pair})

    radius = int(math.ceil(math.sqrt(max_norm_sq))) + 2
    pts: List[Tuple[int, int]] = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            if shell_norm_sq(a, b) in shell_norms:
                pts.append((a, b))
    # Deterministic order: by shell, then by phase-pair sector, then coordinate.
    pts.sort(key=lambda p: (shell_norm_sq(*p), phase_pair_sector(*p), p))
    ab = np.array(pts, dtype=np.int64)
    m = ab.shape[0]
    raw = lattice_array_to_complex(ab)
    sum_norm_sq = int(sum(shell_norm_sq(int(a), int(b)) for a, b in ab))
    scale_sq = Fraction(m, sum_norm_sq)
    scale = math.sqrt(float(scale_sq))
    points_unit = raw * scale

    # ---- verify phase-pair uniformity: every shell has equal per-sector count ----
    index_of = {(int(a), int(b)): i for i, (a, b) in enumerate(ab)}
    phase_pair_uniform = True
    for n in shell_norms:
        per_sector: Dict[int, int] = {k: 0 for k in range(6)}
        for (a, b), _i in index_of.items():
            if shell_norm_sq(a, b) == n:
                per_sector[phase_pair_sector(a, b)] += 1
        if len(set(per_sector.values())) != 1:
            phase_pair_uniform = False
            break

    # ---- build the exact inversion-dual point permutation + verify it pairs ----
    # iota_r maps a point of shell N, sector s to the unique same-sector point of
    # shell r^4/N (k=1 shells -> unique). This is an exact lattice-point
    # permutation (the firewall: a *label* map, not a distance).
    dual_index = np.full(m, -1, dtype=np.int64)
    inversion_paired = True
    for (a, b), i in index_of.items():
        n = shell_norm_sq(a, b)
        s = phase_pair_sector(a, b)
        nd = dual_shell_norm(n, r_sq)
        if nd is None:
            inversion_paired = False
            break
        match = [index_of[(c, d)] for (c, d) in index_of
                 if shell_norm_sq(c, d) == nd and phase_pair_sector(c, d) == s]
        if len(match) != 1:
            inversion_paired = False
            break
        dual_index[i] = match[0]
    if inversion_paired:
        # confirm involution: applying the dual twice is the identity.
        inversion_paired = bool(np.array_equal(dual_index[dual_index],
                                               np.arange(m, dtype=np.int64)))

    fundamental = [i for i, (a, b) in enumerate(ab)
                   if phase_pair_sector(int(a), int(b)) == 0
                   and shell_norm_sq(int(a), int(b)) <= r_sq]

    labels = np.arange(m, dtype=np.int64)  # placeholder; unused for BER
    return Constellation(
        points_unit=points_unit, labels=labels, bits_per_symbol=0, scale=scale,
        name=f"radial-dual-r{r_sq}-Nle{max_norm_sq}", ab=ab,
        scale_sq_exact=scale_sq, shell_norms=tuple(shell_norms),
        inversion_r_sq=r_sq, phase_pair_uniform=phase_pair_uniform,
        inversion_paired=inversion_paired,
        fundamental_domain_size=len(fundamental),
        inversion_dual_index=dual_index,
    )


def _gray(n: int) -> int:
    return n ^ (n >> 1)


def build_square_qam(m: int) -> Constellation:
    """Build a Gray-mapped, unit-average-energy square M-QAM constellation.

    M must be an even power of two (16, 64, 256, ...) so that sqrt(M) is an
    integer side length. Per-axis amplitudes are the odd integers
    {-(L-1), ..., -1, 1, ..., L-1}; the per-axis index is Gray-coded and the two
    axes are concatenated (I in the high bits, Q in the low bits).
    """
    nbits = int(round(math.log2(m)))
    if (1 << nbits) != m or nbits % 2 != 0:
        raise ValueError("square QAM requires M an even power of two (16, 64, 256, ...)")
    side = int(round(math.sqrt(m)))
    half_bits = nbits // 2
    amps = np.array([2 * i - (side - 1) for i in range(side)], dtype=np.float64)
    avg_energy = float(np.mean(amps ** 2)) * 2.0  # I and Q contribute equally
    scale = 1.0 / math.sqrt(avg_energy)

    points = np.empty(m, dtype=np.complex128)
    labels = np.empty(m, dtype=np.int64)
    for i in range(side):          # in-phase index
        for q in range(side):      # quadrature index
            idx = i * side + q
            points[idx] = complex(amps[i], amps[q]) * scale
            labels[idx] = (_gray(i) << half_bits) | _gray(q)
    return Constellation(points_unit=points, labels=labels,
                         bits_per_symbol=nbits, scale=scale, name=f"sqQAM-{m}")


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def decode_ml(received_unit: np.ndarray, points_unit: np.ndarray,
              chunk: int = 20000) -> np.ndarray:
    """Exhaustive maximum-likelihood (minimum-Euclidean-distance) decoder.

    Returns the index into ``points_unit`` of the nearest constellation point for
    each received symbol. O(M) per symbol. Memory is bounded by processing the
    received stream in chunks. This is the ground-truth baseline for both the
    hexagonal and square constellations.
    """
    received_unit = np.asarray(received_unit, dtype=np.complex128)
    n = received_unit.shape[0]
    out = np.empty(n, dtype=np.int64)
    pts = points_unit[np.newaxis, :]
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        seg = received_unit[start:stop, np.newaxis]
        d2 = np.abs(seg - pts) ** 2
        out[start:stop] = np.argmin(d2, axis=1)
    return out


@dataclass
class _HexDecodeContext:
    """Precomputed structures that make the closed-form hex decoder O(1)/symbol."""
    constellation: Constellation
    amin: int
    bmin: int
    index_grid: np.ndarray  # int grid; -1 where no constellation point sits


def make_hex_decode_context(constellation: Constellation) -> _HexDecodeContext:
    """Precompute the integer occupancy grid mapping lattice (a, b) -> point index."""
    ab = constellation.ab
    amin, amax = int(ab[:, 0].min()), int(ab[:, 0].max())
    bmin, bmax = int(ab[:, 1].min()), int(ab[:, 1].max())
    grid = np.full((amax - amin + 1, bmax - bmin + 1), -1, dtype=np.int64)
    grid[ab[:, 0] - amin, ab[:, 1] - bmin] = np.arange(ab.shape[0], dtype=np.int64)
    return _HexDecodeContext(constellation, amin, bmin, grid)


def nearest_lattice_point(received_unit: np.ndarray,
                          scale: float) -> np.ndarray:
    """Closed-form A2 nearest-(infinite-)lattice-point search; O(1) per symbol.

    The received unit-energy symbols are de-scaled to lattice space (z = y / s),
    expressed in oblique coordinates, rounded, and compared against the 3x3
    window of integer candidates around the rounded point. The hexagonal Voronoi
    cell is contained in that window (the lattice covering radius forbids a
    nearer point two cells away), so the window minimum is the exact nearest
    lattice point. Returns an (N, 2) integer array of (a, b).
    """
    z = np.asarray(received_unit, dtype=np.complex128) / scale
    a_real, b_real = complex_to_oblique(z)
    na = np.round(a_real).astype(np.int64)
    nb = np.round(b_real).astype(np.int64)

    best_d2 = np.full(z.shape, np.inf)
    best_a = na.copy()
    best_b = nb.copy()
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            ca = na + da
            cb = nb + db
            cz = (ca + cb * _OMEGA1_RE) + 1j * (cb * _OMEGA1_IM)
            d2 = np.abs(z - cz) ** 2
            better = d2 < best_d2
            best_d2 = np.where(better, d2, best_d2)
            best_a = np.where(better, ca, best_a)
            best_b = np.where(better, cb, best_b)
    return np.stack([best_a, best_b], axis=1)


def decode_hex_fast(received_unit: np.ndarray,
                    ctx: _HexDecodeContext) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-time hexagonal demodulator, bitwise-identical to exhaustive ML.

    Steps: (1) find the exact nearest infinite-lattice point via the O(1) window;
    (2) if that point belongs to the finite constellation, accept it -- this is
    the ML decision, since the global nearest lattice point lying inside the
    constellation is also the nearest among the constellation; (3) otherwise the
    received symbol lies outside the constellation region and we fall back to an
    exhaustive nearest-constellation-point search for those (rare, at usable SNR)
    symbols. The fallback guarantees the output equals ML for every symbol.

    Complexity: O(1) per symbol on the fast (interior) path, which covers the
    large majority of symbols at usable SNR; the exterior fallback is O(M), so
    the worst case is O(M) and the *amortized* cost depends on the exterior
    fraction (1 - mean(fast_path)), which grows as SNR falls or M grows. Report
    fast_path alongside any throughput number so the O(1) claim is qualified by
    the fraction of symbols that actually took the constant-time path.

    Returns
    -------
    indices:
        (N,) indices into the constellation points.
    fast_path:
        (N,) boolean mask, True where the O(1) window result was used directly
        (i.e., no exhaustive fallback). Reported by the experiments to show that
        the constant-time path covers essentially all symbols.
    """
    con = ctx.constellation
    ab = nearest_lattice_point(received_unit, con.scale)
    n = ab.shape[0]

    rows = ab[:, 0] - ctx.amin
    cols = ab[:, 1] - ctx.bmin
    in_bounds = ((rows >= 0) & (rows < ctx.index_grid.shape[0]) &
                 (cols >= 0) & (cols < ctx.index_grid.shape[1]))
    indices = np.full(n, -1, dtype=np.int64)
    if np.any(in_bounds):
        gi = ctx.index_grid[rows[in_bounds], cols[in_bounds]]
        tmp = indices[in_bounds]
        tmp[:] = gi
        indices[in_bounds] = tmp
    fast_path = indices >= 0
    if not np.all(fast_path):
        miss = ~fast_path
        indices[miss] = decode_ml(np.asarray(received_unit)[miss], con.points_unit)
    return indices, fast_path


# ---------------------------------------------------------------------------
# Phase-pair + inversion FOLDED decoder (shell-complete radial-dual constellation)
# ---------------------------------------------------------------------------
#
# For a shell-complete, phase-pair-uniform, inversion-paired constellation (the
# C7 radial-dual object), the membership/label structure folds to the
# fundamental domain while the decision stays exact ML. The fold is purely a
# storage/label operation -- the inversion firewall: the Euclidean nearest-point
# test below uses true distances only; inversion never touches a metric.
#
# Indexing convention (set by build_radial_dual_constellation): points are
# ordered by (shell norm, phase-pair sector), and each complete shell holds
# exactly one point per sector, so the full index of a point equals
#     6 * rank(shell) + sector.
# That lets the decoder reconstruct any full index from a tiny stored table:
#   * phase-pair fold: store one representative per shell  (len = #shells);
#   * phase-pair + inversion fold: store only inner/boundary shells
#     (len = #inner shells) and recover an outer shell from its integer dual.

def _reconstruct_full_shells(stored_inner: Sequence[int], r_sq: int) -> Tuple[int, ...]:
    """Rebuild the full shell set from the stored inner shells + r_sq alone.

    Each stored inner/boundary shell ``n`` implies its outer dual ``r^4/n``; the
    union is the complete shell set. This is what makes the inversion fold a
    genuine storage reduction: the outer half of the membership table need not be
    stored, only regenerated from the inner half.
    """
    full = set(int(n) for n in stored_inner)
    r4 = r_sq * r_sq
    for n in list(full):
        if n != 0 and r4 % n == 0:
            full.add(r4 // n)
    return tuple(sorted(full))


@dataclass
class _FoldedDecodeContext:
    """Folded membership/label table for the radial-dual constellation decoder."""
    constellation: Constellation
    r_sq: int
    fold_inversion: bool
    stored_shells: Tuple[int, ...]   # shell norms physically stored (the fold)
    full_shells: Tuple[int, ...]     # full shell set (reconstructed when folded)
    stored_table_size: int           # = len(stored_shells); the storage metric


def make_folded_decode_context(constellation: Constellation,
                               fold_inversion: bool = True) -> _FoldedDecodeContext:
    """Precompute the folded decode table for a radial-dual constellation.

    With ``fold_inversion=False`` the table keeps one representative per complete
    shell (the phase-pair / rotation fold). With ``fold_inversion=True`` it keeps
    only the inner + boundary shells (the combined phase-pair + inversion fold)
    and regenerates the outer shells from their exact integer duals. The reported
    ``stored_table_size`` is the count of stored shells -- the quantity that
    shrinks (e.g. 7 -> 4 for the r^2 = 12 constellation), since the six per-shell
    sector points are regenerated by rotation rather than stored.
    """
    if constellation.shell_norms is None or constellation.inversion_r_sq is None:
        raise ValueError("folded decode requires a radial-dual constellation "
                         "(build_radial_dual_constellation)")
    r_sq = int(constellation.inversion_r_sq)
    full_shells = tuple(int(n) for n in constellation.shell_norms)
    if fold_inversion:
        stored = tuple(n for n in full_shells if n <= r_sq)        # inner + boundary
        rebuilt = _reconstruct_full_shells(stored, r_sq)
        # Internal consistency: the inner half must regenerate the full shell set.
        if rebuilt != full_shells:
            raise ValueError("inner shells do not regenerate the full shell set; "
                             "constellation is not inversion-complete")
    else:
        stored = full_shells                                        # all shells
    return _FoldedDecodeContext(constellation, r_sq, fold_inversion,
                                stored, full_shells, len(stored))


def decode_hex_folded(received_unit: np.ndarray,
                      ctx: _FoldedDecodeContext) -> Tuple[np.ndarray, np.ndarray]:
    """Exact ML decode for a radial-dual constellation using only the folded table.

    Bitwise-identical to exhaustive ML. The fast path fires when the true nearest
    lattice point is itself a constellation point (then it is the ML point); its
    full index is reconstructed from the folded table via the phase-pair sector
    (rotation) and, for outer shells, the exact integer shell dual (inversion).
    Symbols whose nearest lattice point falls in a radial gap take the exhaustive
    ML fallback over the full point set (true distances). Returns
    ``(indices, fast_path_mask)``.

    The decision is purely Euclidean (the nearest-lattice-point step); inversion
    enters only to regenerate stored labels (the firewall), so exactness is
    preserved regardless of the fold.
    """
    con = ctx.constellation
    r_sq = ctx.r_sq
    r4 = r_sq * r_sq
    ab = nearest_lattice_point(received_unit, con.scale)
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    n = a * a + a * b + b * b                       # shell norm of each q*
    sec = phase_pair_sector_array(ab)

    stored = set(ctx.stored_shells)
    # Fold each shell to its fundamental (stored) representative shell.
    if ctx.fold_inversion:
        is_inner = n <= r_sq
        divisible = (n > 0) & (r4 % np.where(n > 0, n, 1) == 0)
        fold_n = np.where(is_inner, n, np.where(divisible, r4 // np.where(n > 0, n, 1), -1))
    else:
        fold_n = n
    member = np.array([(int(fn) in stored) for fn in fold_n], dtype=bool) & (sec >= 0)

    # Reconstruct full index = 6 * rank(shell) + sector, ranks from the full set.
    full_sorted = np.array(ctx.full_shells, dtype=np.int64)
    indices = np.full(n.shape, -1, dtype=np.int64)
    if np.any(member):
        nm = n[member]
        rank = np.searchsorted(full_sorted, nm)
        indices[member] = 6 * rank + sec[member]
    fast_path = member
    if not np.all(fast_path):
        miss = ~fast_path
        indices[miss] = decode_ml(np.asarray(received_unit)[miss], con.points_unit)
    return indices, fast_path


# ---------------------------------------------------------------------------
# Differential hexagonal (senary) sector coding -- rotation-robustness study
# ---------------------------------------------------------------------------

def differential_encode(sector_data: np.ndarray) -> np.ndarray:
    """Differentially encode a stream of senary symbols d_n in {0..5}.

    Transmitted sector index s_n = (s_{n-1} + d_n) mod 6, with s_{-1} = 0. A
    static carrier-phase ambiguity equal to a multiple of pi/3 shifts every s_n
    by the same constant, which cancels in the receiver's consecutive
    differences -- the hexagonal analogue of DPSK.
    """
    sector_data = np.asarray(sector_data, dtype=np.int64) % 6
    return np.cumsum(sector_data) % 6


def differential_decode(received_sectors: np.ndarray) -> np.ndarray:
    """Recover the senary data from received sector indices via consecutive
    differences (mod 6), assuming an initial reference sector of 0."""
    received_sectors = np.asarray(received_sectors, dtype=np.int64) % 6
    prev = np.empty_like(received_sectors)
    prev[0] = 0
    prev[1:] = received_sectors[:-1]
    return (received_sectors - prev) % 6


# ---------------------------------------------------------------------------
# Combined rotation + inversion (C6 x Z2) differential coding -- the T24 codec
# ---------------------------------------------------------------------------
#
# The plain differential codec above absorbs a static carrier-phase ambiguity
# (a multiple of pi/3, i.e. an element of the order-6 rotation group C6). The
# T24 codec additionally absorbs a static amplitude-inversion ambiguity (an
# element of the order-2 radial inversion group Z2), so it is invariant under
# the full order-12 rotation x inversion group C6 x Z2 -- the operationally
# relevant rotation-and-inversion subgroup of the centrosymmetric hexagonal
# point group D_6h. The carried state is a (sector in Z6, inversion bit in Z2)
# pair; data is transmitted as consecutive differences of each component, so a
# constant offset applied to the whole stream (any of the 12 static actions)
# cancels in the receiver's differences. This is a label-domain construction:
# the inversion bit is a discrete state, never a Euclidean operation.

def differential_encode_t24(d_sector: np.ndarray,
                            d_inversion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Differentially encode paired senary/binary data (d_sec in {0..5},
    d_inv in {0,1}) into transmitted (sector, inversion-state) streams.

    Transmitted state s_n = (s_{n-1} + d_sector_n) mod 6 and
    u_n = (u_{n-1} + d_inversion_n) mod 2, with s_{-1} = u_{-1} = 0. A static
    rotation by k*pi/3 shifts every s_n by k; a static amplitude inversion flips
    every u_n by 1. Both are constant offsets that cancel in the receiver's
    consecutive differences, so the codec is invariant under all 12 elements of
    the combined rotation x inversion group C6 x Z2.

    Returns the (sector_stream, inversion_stream) pair to transmit.
    """
    d_sector = np.asarray(d_sector, dtype=np.int64) % 6
    d_inversion = np.asarray(d_inversion, dtype=np.int64) % 2
    s = np.cumsum(d_sector) % 6
    u = np.cumsum(d_inversion) % 2
    return s, u


def differential_decode_t24(sector_stream: np.ndarray,
                            inversion_stream: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Recover paired (d_sector, d_inversion) data from received (sector,
    inversion-state) streams via component-wise consecutive differences
    (mod 6 and mod 2), assuming an initial reference state (0, 0).

    Any static C6 x Z2 action applied uniformly to both streams (a constant
    sector offset and/or a global inversion flip) cancels in the differences,
    leaving the data unchanged from the second symbol onward.
    """
    s = np.asarray(sector_stream, dtype=np.int64) % 6
    u = np.asarray(inversion_stream, dtype=np.int64) % 2
    ps = np.empty_like(s)
    ps[0] = 0
    ps[1:] = s[:-1]
    pu = np.empty_like(u)
    pu[0] = 0
    pu[1:] = u[:-1]
    return (s - ps) % 6, (u - pu) % 2


# ---------------------------------------------------------------------------
# Channels (unit-average-energy input symbols)
# ---------------------------------------------------------------------------

def _noise_sigma_sq(ebn0_db: float, bits_per_symbol: int) -> float:
    """Total complex-noise variance sigma^2 = N0 for unit symbol energy Es = 1.

    Es/N0 (dB) = Eb/N0 (dB) + 10*log10(bits/symbol); N0 = Es / (Es/N0)_linear.
    """
    es_n0_db = ebn0_db + 10.0 * math.log10(max(bits_per_symbol, 1))
    es_n0_lin = 10.0 ** (es_n0_db / 10.0)
    return 1.0 / es_n0_lin


def _complex_gaussian(n: int, sigma_sq: float, rng: np.random.Generator) -> np.ndarray:
    """n i.i.d. CN(0, sigma_sq) samples (variance sigma_sq/2 per real component)."""
    s = math.sqrt(sigma_sq / 2.0)
    return rng.normal(0.0, s, n) + 1j * rng.normal(0.0, s, n)


def awgn(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int,
         rng: np.random.Generator) -> np.ndarray:
    """Add complex AWGN at the given Eb/N0 to unit-energy symbols."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    sigma_sq = _noise_sigma_sq(ebn0_db, bits_per_symbol)
    return symbols + _complex_gaussian(symbols.shape[0], sigma_sq, rng)


def impulsive(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int,
              rng: np.random.Generator, p: float = 0.1,
              amplitude: float = 5.0) -> np.ndarray:
    """Add 2D impulsive noise: background AWGN, plus with probability p a symbol
    is struck by an impulse of magnitude ``amplitude`` at a uniformly random
    phase (generalizing the BPSK case study's +/-A outliers to the plane)."""
    symbols = np.asarray(symbols, dtype=np.complex128)
    n = symbols.shape[0]
    sigma_sq = _noise_sigma_sq(ebn0_db, bits_per_symbol)
    out = symbols + _complex_gaussian(n, sigma_sq, rng)
    hit = rng.random(n) < p
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    out[hit] += amplitude * np.exp(1j * theta[hit])
    return out


def rayleigh(symbols: np.ndarray, ebn0_db: float, bits_per_symbol: int,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Flat Rayleigh fading y = h*x + n with h ~ CN(0, 1) (E|h|^2 = 1).

    Returns (received y, gains h). With perfect CSI the receiver decodes y / h
    (zero forcing), which keeps the nearest-point decoder exact; the average SNR
    matches the AWGN mapping because E|h|^2 = 1.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    n = symbols.shape[0]
    sigma_sq = _noise_sigma_sq(ebn0_db, bits_per_symbol)
    h = _complex_gaussian(n, 1.0, rng)            # E|h|^2 = 1
    y = h * symbols + _complex_gaussian(n, sigma_sq, rng)
    return y, h


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def clopper_pearson(errors: int, trials: int,
                    alpha: float = 0.05) -> Tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided 1-alpha binomial confidence interval.

    Returns (lower, upper) for the error probability given ``errors`` out of
    ``trials``. Handles the boundary cases errors == 0 and errors == trials.
    """
    if trials <= 0:
        return (0.0, 1.0)
    if errors <= 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2.0, errors, trials - errors + 1)
    if errors >= trials:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1.0 - alpha / 2.0, errors + 1, trials - errors)
    return (float(lo), float(hi))


def hamming_bits(a: np.ndarray, b: np.ndarray, nbits: int) -> int:
    """Total number of differing bits between two integer-label arrays."""
    x = np.bitwise_xor(a.astype(np.int64), b.astype(np.int64))
    total = 0
    for _ in range(nbits):
        total += int(np.sum(x & 1))
        x >>= 1
    return total


def collect_provenance(device=None) -> Dict[str, object]:
    """Capture the hardware/version facts the paper's Methods table needs, and
    -- for the GPU study -- certify *where* a run executed.

    ``device`` is an optional torch device (only Simulation 04 passes one). The
    decisive field is ``ran_on_cuda``. Supporting fields disambiguate the two
    non-GPU cases that otherwise look identical in a results table:
      * ``torch_built_with_cuda is None``  -> a CPU-only torch wheel; this
        install can NEVER use a GPU, regardless of hardware present.
      * ``torch_built_with_cuda`` set but ``cuda_is_available is False`` -> a
        CUDA-capable torch build that found no usable GPU at runtime.
    """
    prov: Dict[str, object] = {
        "tqf_version": __version__,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": getattr(stats, "__version__", None) or _scipy_version(),
        "torch_installed": False,
        "torch_version": None,
        "torch_built_with_cuda": None,   # torch.version.cuda; None on CPU-only wheels
        "cudnn_version": None,
        "cuda_is_available": False,
        "cuda_device_count": 0,
        "device_used": (device.type if device is not None else "n/a"),
        "gpu_name": None,
        "gpu_compute_capability": None,
        "gpu_total_mem_gb": None,
        "torch_num_threads": None,
        "ran_on_cuda": False,
    }
    try:
        import torch
    except ImportError:
        return prov
    prov["torch_installed"] = True
    prov["torch_version"] = torch.__version__
    prov["torch_built_with_cuda"] = torch.version.cuda
    prov["torch_num_threads"] = torch.get_num_threads()
    try:
        prov["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    avail = bool(torch.cuda.is_available())
    prov["cuda_is_available"] = avail
    if avail:
        prov["cuda_device_count"] = torch.cuda.device_count()
        try:
            props = torch.cuda.get_device_properties(0)
            prov["gpu_name"] = props.name
            prov["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            prov["gpu_total_mem_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            prov["gpu_name"] = torch.cuda.get_device_name(0)
    prov["ran_on_cuda"] = bool(avail and device is not None and device.type == "cuda")
    return prov


def _scipy_version() -> str:
    try:
        import scipy
        return scipy.__version__
    except Exception:
        return "unknown"


def emit_provenance(results_dir: str, sim_name: str, device=None,
                    args=None, extra=None) -> Dict[str, object]:
    """Print a compact provenance header and write ``<sim_name>_provenance.json``
    next to the results, so every study self-documents its environment. Returns
    the provenance dict for inline use (e.g. Simulation 04's ran_on_cuda gate).
    """
    prov = collect_provenance(device)
    print(f"  env: python {prov['python_version']}  numpy {prov['numpy_version']}  "
          f"scipy {prov['scipy_version']}  | {prov['platform']}")
    if prov["torch_installed"]:
        print(f"  torch {prov['torch_version']}  cuda_build={prov['torch_built_with_cuda']}"
              f"  cuda_available={prov['cuda_is_available']}  gpu={prov['gpu_name']}")
    os.makedirs(results_dir, exist_ok=True)
    payload = {"provenance": prov}
    if args is not None:
        payload["args"] = vars(args) if hasattr(args, "__dict__") else dict(args)
    if extra is not None:
        payload["extra"] = extra
    path = os.path.join(results_dir, f"{sim_name}_provenance.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return prov


def mcnemar_pvalue(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value for a paired binary comparison.

    ``n01`` and ``n10`` are the two *discordant* cell counts -- the number of
    paired trials in which exactly one of the two methods errs (here: square errs
    while hex is correct, and hex errs while square is correct, respectively).
    Under the null hypothesis that the two methods are equally likely to be the
    sole error, the smaller discordant count is Binomial(n01 + n10, 1/2), and the
    exact two-sided p-value follows. Returns 1.0 when there are no discordant
    pairs. The statistic is symmetric in its two arguments; the *direction* of any
    resolved difference is decided by the caller from which count is larger.

    This is the correct per-point test for the common-random-number (paired)
    hex-vs-square stream: it conditions on the discordant pairs and so exploits
    the pairing, unlike a comparison of the two marginal Clopper-Pearson bands.
    """
    n01 = int(n01)
    n10 = int(n10)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    return float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down multiple-comparison correction.

    Given a family of ``pvalues``, control the family-wise error rate at
    ``alpha`` while being uniformly more powerful than plain Bonferroni. Returns
    ``(reject, p_adjusted)`` as arrays aligned with the *input* order:
    ``reject[i]`` is True iff hypothesis ``i`` is rejected at level ``alpha``, and
    ``p_adjusted[i]`` is its monotone Holm-adjusted p-value, so that
    ``reject == (p_adjusted <= alpha)`` exactly. An empty family yields empty
    arrays; a single hypothesis is returned uncorrected.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    # Step-down adjusted p-values: running maximum of (m - i) * p_(i), capped at
    # 1 so the sequence is monotone non-decreasing in the sorted order.
    adj_sorted = np.maximum.accumulate(
        np.clip((m - np.arange(m)) * p_sorted, 0.0, 1.0))
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    return p_adj <= alpha, p_adj


if __name__ == "__main__":
    # Minimal smoke test of the library's core invariants.
    print(f"tqf_hex_signal version {__version__}")
    con = build_filled_constellation(64)
    print(f"built {con.name}: size={con.size}, bits/symbol={con.bits_per_symbol}, "
          f"avg energy={np.mean(np.abs(con.points_unit) ** 2):.6f}")
    ctx = make_hex_decode_context(con)
    rng = np.random.default_rng(0)
    tx_idx = rng.integers(0, con.size, 5000)
    rx = awgn(con.points_unit[tx_idx], ebn0_db=12.0,
              bits_per_symbol=con.bits_per_symbol, rng=rng)
    dec_fast, fast_mask = decode_hex_fast(rx, ctx)
    dec_ml = decode_ml(rx, con.points_unit)
    print(f"fast==ML on {np.mean(dec_fast == dec_ml) * 100:.4f}% of symbols; "
          f"O(1) fast path used on {np.mean(fast_mask) * 100:.2f}%")
