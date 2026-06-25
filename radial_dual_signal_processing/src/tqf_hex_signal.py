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
  * Channels generalizing the BPSK case study to 2-D: complex AWGN, 2-D
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
Version: 1.0.0
Date: June 24, 2026
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


def sector_index(a: int, b: int) -> int:
    """Return the angular sector index in {0, ..., 5} of a non-origin point (a, b).

    Sector S_k is the half-open 60 degree wedge [d_k, d_{k+1}); a point lying
    exactly on primary ray d_k is assigned to sector k. The test uses only
    integer determinants (no floating-point phase), so it is exact. The origin
    has no defined sector and returns -1.
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


def sector_index_array(ab: np.ndarray) -> np.ndarray:
    """Vectorized exact sector index for an (N, 2) integer array; origin -> -1."""
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
    """Add 2-D impulsive noise: background AWGN, plus with probability p a symbol
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
