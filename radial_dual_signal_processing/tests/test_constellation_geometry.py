"""
test_constellation_geometry.py - Exact Constellation Geometry, Fairness, and Geometry-Price Tests

Topic-focused tests for constellation construction and the exact integer/rational
geometry machinery:

  * build_filled_constellation_any: agreement with the power-of-two builder on
    the shared point set, and the exact hex-42 baseline properties;
  * the Study 3 exact constellation geometry (d_min^2, nearest-neighbor
    multiplicity, PAPR, Gray-map Hamming) for hexagonal and square QAM, and the
    nearest-neighbor-approximation gain prediction;
  * the Study 8 exact geometry of the radial-dual / hex-42 pair, the common-
    random-number pairing, the exact-decoder tie-in, and the nearest-neighbor
    price that anchors the pre-registered bracket.

The helpers are imported from the shipped simulation modules, so these tests
validate the actual code that produces the paper's numbers.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.2.0
Date: July 4, 2026
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import tqf_hex_signal as t
import simulation_03_symmetry_reduced_metric_exact as sim03
import simulation_08_radial_dual_geometry_price as sim08


# --------------------------------------------------------------------------- #
# build_filled_constellation_any
# --------------------------------------------------------------------------- #
def test_build_filled_any_matches_pow2_point_set():
    # For a power-of-two m the selected lattice points and the exact scale agree
    # with the power-of-two builder (only labels / bits differ).
    m = 16
    any_con = t.build_filled_constellation_any(m)
    pow2 = t.build_filled_constellation(m)
    assert any_con.size == pow2.size == m
    assert any_con.scale_sq_exact == pow2.scale_sq_exact
    any_pts = {(int(a), int(b)) for a, b in any_con.ab}
    pow2_pts = {(int(a), int(b)) for a, b in pow2.ab}
    assert any_pts == pow2_pts
    assert any_con.bits_per_symbol == 0            # label-free


def test_hex_any_42_properties():
    con = t.build_filled_constellation_any(42)
    assert con.size == 42
    assert con.name == "hex-any-42"
    assert con.bits_per_symbol == 0
    assert con.scale_sq_exact == Fraction(7, 41)
    # origin is included (minimum-energy set)
    assert any(int(a) == 0 and int(b) == 0 for a, b in con.ab)


def test_build_filled_any_rejects_tiny():
    with pytest.raises(ValueError):
        t.build_filled_constellation_any(1)

# --------------------------------------------------------------------------- #
# Study 3 exact constellation-geometry values (imported from the sim module)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,d2,nn,papr,gray", [
    (16, Fraction(4, 9), 66, Fraction(16, 9), Fraction(19, 11)),
    (64, Fraction(64, 567), 326, Fraction(1216, 567), Fraction(342, 163)),
    (256, Fraction(256, 9027), 1422, Fraction(18688, 9027), Fraction(1726, 711)),
])
def test_hex_geometry_exact(m, d2, nn, papr, gray):
    con = t.build_filled_constellation(m)
    gd2, gnn, gpapr, d_lat = sim03._exact_lattice_geometry(con)
    assert gd2 == d2
    assert gnn == nn
    assert gpapr == papr
    assert sim03._hex_gray_nn_hamming(con, d_lat) == gray


@pytest.mark.parametrize("m,d2,nn,papr", [
    (16, Fraction(2, 5), 48, Fraction(9, 5)),
    (64, Fraction(2, 21), 224, Fraction(7, 3)),
    (256, Fraction(2, 85), 960, Fraction(45, 17)),
])
def test_square_geometry_exact(m, d2, nn, papr):
    gd2, gnn, gpapr, gray = sim03._exact_square_geometry(m)
    assert gd2 == d2
    assert gnn == nn
    assert gpapr == papr
    assert gray == Fraction(1)                     # true Gray: 1 bit per NN step


def test_nn_prediction_m64_matches_plan():
    # Pre-registered: M=64 @ SER 1e-2 predicted hex-vs-square gain ~ +0.368 dB.
    h = sim03._exact_lattice_geometry(t.build_filled_constellation(64))
    q = sim03._exact_square_geometry(64)
    bits = math.log2(64)
    eh = sim03._nn_predicted_ebn0(float(h[0]), h[1] / 64, bits, 1e-2)
    eq = sim03._nn_predicted_ebn0(float(q[0]), q[1] / 64, bits, 1e-2)
    assert (eq - eh) == pytest.approx(0.368, abs=0.01)

# --------------------------------------------------------------------------- #
# Study 8 exact geometry, CRN pairing, C1 tie-in, NN-approximation price
# --------------------------------------------------------------------------- #
def test_sim08_radial_dual_geometry_exact():
    rd = t.build_radial_dual_constellation(12, 60)
    d2, nn, papr = sim08._exact_geometry(rd)
    assert d2 == Fraction(7, 128)
    assert nn == 48
    assert papr == Fraction(21, 8)


def test_sim08_hex42_geometry_exact():
    hx = t.build_filled_constellation_any(42)
    d2, nn, papr = sim08._exact_geometry(hx)
    assert d2 == Fraction(7, 41)
    assert nn == 200
    assert papr == Fraction(84, 41)


def test_sim08_crn_rewind_gives_identical_noise():
    # The RNG-state rewind must feed BOTH constellations the identical complex
    # noise realization at a grid point (common random numbers).
    con = t.build_radial_dual_constellation(12, 60)
    rng = np.random.default_rng(42)
    sym = con.points_unit[rng.integers(0, con.size, 5000)]
    state = rng.bit_generator.state
    rx_a = t.awgn(sym, 14.0, 0, rng)
    rng.bit_generator.state = state
    rx_b = t.awgn(sym, 14.0, 0, rng)
    assert np.array_equal(rx_a, rx_b)


def test_sim08_sweep_c1_tie_in_and_direction():
    rd = t.build_radial_dual_constellation(12, 60)
    hx = t.build_filled_constellation_any(rd.size)
    rng = np.random.default_rng(42)
    rows, ml_ok = sim08.run_sweep(rd, hx, [10.0, 16.0, 22.0], 8000, 0.05, rng)
    # C1 tie-in: both practical decoders equal ML at every point.
    assert ml_ok
    for r in rows:
        assert r["rd_decoder_matches_ml"] == 1.0
        assert r["hex42_decoder_matches_ml"] == 1.0
    # At high SNR the small-d_min radial-dual constellation is the worse one.
    assert rows[-1]["rd_ser"] >= rows[-1]["hex42_ser"]


def test_sim08_nn_predicted_price_anchors_bracket():
    # The pre-registered bracket is centered on this model price.
    price = sim08._nn_predicted_price(1e-2)
    assert price == pytest.approx(3.33, abs=0.1)
    lo, hi = sim08._PRICE_BRACKET_DB
    assert lo <= price <= hi
