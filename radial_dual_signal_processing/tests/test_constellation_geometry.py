"""
test_constellation_geometry.py - Exact Constellation Geometry, Fairness, and Geometry-Price Tests

Topic-focused tests for constellation construction and the exact integer/rational
geometry machinery:

  * build_filled_constellation_any: agreement with the power-of-two builder on
    the shared point set, and the exact hex-42 baseline properties;
  * the Study 3 exact constellation geometry (exact squared-distance enumerator,
    lattice d_min^2, nearest-neighbor multiplicity) and the hexagonal packing
    advantage over square QAM at matched energy (C3 / C4);
  * the Study 6 radial-dual geometry price: the exact structure of the radial-dual
    / hex-42 pair, the (d_min, N_nn) crossover DIRECTION prediction, the common-
    random-number paired AWGN sweep, and the low-/high-SNR sign flip (C7 / C9).

The helpers are imported from the shipped library and simulation modules, so
these tests validate the actual code that produces the paper's numbers.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 8, 2026
"""

from fractions import Fraction

import numpy as np
import pytest

import tqf_hex_signal as t
import simulation_03_symmetry_reduced_metric as sim03
import simulation_06_radial_dual_geometry_price as sim06


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
# Study 3 exact constellation geometry (via the shipped exact enumerator)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [16, 64, 256])
def test_hex_exact_min_distance_and_multiplicity(m):
    # The exact integer squared-distance enumerator of the hex constellation puts
    # its nearest neighbors at unit squared LATTICE distance, and the exact
    # signal-space d_min^2 is the rational scale times that integer.
    con = t.build_filled_constellation(m)
    full = sim03._full_enumerator(con.ab.astype(np.int64))
    d2_lat = min(full.keys())
    assert d2_lat == 1                              # hex nearest neighbors: unit lattice step
    mult = full[d2_lat]
    assert mult > 0 and mult % 2 == 0              # ordered pairs come in symmetric pairs
    d2_signal = con.scale_sq_exact * d2_lat
    assert isinstance(d2_signal, Fraction) and d2_signal > 0


@pytest.mark.parametrize("m", [16, 64, 256])
def test_hex_packs_tighter_than_square_at_matched_energy(m):
    # C3 packing precondition: at matched order and unit average energy the
    # hexagonal constellation has a strictly larger minimum distance than square
    # QAM (the geometric source of the packing gain).
    hexd = sim06._dmin_and_nn(t.build_filled_constellation(m))[0]
    sqd = sim06._dmin_and_nn(t.build_square_qam(m))[0]
    assert hexd > sqd


# --------------------------------------------------------------------------- #
# Study 6 radial-dual geometry, crossover direction, and paired price (C7 / C9)
# --------------------------------------------------------------------------- #
def _rd_hex_pair():
    rd = t.build_radial_dual_constellation(48, 192)     # Study 6 canonical M=42 object
    fl = t.build_filled_constellation_any(rd.size)
    return rd, fl


def test_radial_dual_canonical_shape_and_pairing():
    rd, fl = _rd_hex_pair()
    assert rd.size == fl.size == 42
    assert rd.inversion_paired is True
    assert rd.phase_pair_uniform is True


def test_radial_dual_smaller_dmin_and_nn_predicts_crossover():
    # C9: radial-dual has BOTH the smaller mean nearest-neighbor count and the
    # smaller d_min, so the union-bound proxy predicts a genuine crossover
    # (radial-dual better at low SNR, filled better at high SNR).
    rd, fl = _rd_hex_pair()
    d_rd, _, nn_rd = sim06._dmin_and_nn(rd)
    d_fl, _, nn_fl = sim06._dmin_and_nn(fl)
    assert nn_rd < nn_fl
    assert d_rd < d_fl
    pred = sim06._predicted_direction(d_rd, nn_rd, d_fl, nn_fl)
    assert pred["crossover_predicted"] is True


def test_paired_sweep_shows_low_high_snr_sign_flip(monkeypatch):
    # The shipped CRN-paired AWGN sweep must reproduce the predicted DIRECTION:
    # radial-dual at least ties/beats filled at low Es/N0 and loses at high Es/N0.
    monkeypatch.setattr(sim06, "TRIALS", 6000)
    monkeypatch.setattr(sim06, "ESN0_GRID_DB", [0, 12, 30])
    rd, fl = _rd_hex_pair()
    rng = np.random.default_rng(42)
    rows = sim06._paired_sweep(rd, fl, rng)
    assert [r["esn0_db"] for r in rows] == [0, 12, 30]
    assert rows[0]["rd_ser"] <= rows[0]["fl_ser"]      # radial-dual wins low SNR
    assert rows[-1]["fl_ser"] <= rows[-1]["rd_ser"]    # filled wins high SNR


def test_crn_rewind_gives_identical_noise():
    # The RNG-state rewind must feed BOTH constellations the identical complex
    # noise realization at a grid point (common random numbers).
    con = t.build_radial_dual_constellation(48, 192)
    rng = np.random.default_rng(42)
    sym = con.points_unit[rng.integers(0, con.size, 5000)]
    state = rng.bit_generator.state
    rx_a = t.awgn(sym, 14.0, 1.0, rng)
    rng.bit_generator.state = state
    rx_b = t.awgn(sym, 14.0, 1.0, rng)
    assert np.array_equal(rx_a, rx_b)
