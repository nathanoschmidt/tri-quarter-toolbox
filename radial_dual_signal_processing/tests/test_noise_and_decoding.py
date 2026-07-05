"""
test_noise_and_decoding.py - Noise Model, Folded Decoder, and Phase-Offset Decoding Tests

Topic-focused tests for the signal-level primitives:

  * the fractional-bits Eb/N0 -> noise mapping (log2(6) senary case) with the
    integer-bits regression and the zero-bits (label-free -> Es/N0) clamp;
  * the vectorized folded-decoder membership test == a reference Python-loop
    implementation and == exhaustive ML under both folds, and the derived
    stored-shell array;
  * a phase-offset sanity check: the coherent (absolute-sector) decoder slips at
    a 60 deg offset while the differential (relative) decoder returns to floor.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.2.0
Date: July 4, 2026
"""

import math

import numpy as np
import pytest

import tqf_hex_signal as t
import simulation_05_phase_rotation_robustness as sim05


# --------------------------------------------------------------------------- #
# library version smoke check (well-formed, not pinned to a specific release)
# --------------------------------------------------------------------------- #
def test_library_exposes_wellformed_version():
    parts = t.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


# --------------------------------------------------------------------------- #
# fractional-bits noise mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ebn0", [0.0, 6.0, 10.0, 14.0])
def test_noise_sigma_fractional_bits_log2_6(ebn0):
    bits = math.log2(6.0)
    sig2 = t._noise_sigma_sq(ebn0, bits)
    es_n0_lin = 10.0 ** ((ebn0 + 10.0 * math.log10(bits)) / 10.0)
    assert sig2 == pytest.approx(1.0 / es_n0_lin, rel=1e-12)


@pytest.mark.parametrize("k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("ebn0", [0.0, 6.0, 10.0])
def test_noise_sigma_integer_bits_unchanged(ebn0, k):
    # Integer-bits behavior must be unchanged (the existing suite pins it).
    sig2 = t._noise_sigma_sq(ebn0, k)
    es_n0_lin = 10.0 ** ((ebn0 + 10.0 * math.log10(k)) / 10.0)
    assert sig2 == pytest.approx(1.0 / es_n0_lin, rel=1e-12)


@pytest.mark.parametrize("ebn0", [0.0, 7.5, 12.0])
def test_noise_sigma_zero_bits_clamps_to_esn0(ebn0):
    # Label-free constellations pass bits_per_symbol = 0; the knob is Es/N0,
    # i.e. identical to bits = 1.
    assert t._noise_sigma_sq(ebn0, 0) == t._noise_sigma_sq(ebn0, 1)
    assert t._noise_sigma_sq(ebn0, 0.0) == pytest.approx(
        1.0 / (10.0 ** (ebn0 / 10.0)), rel=1e-12)

# --------------------------------------------------------------------------- #
# vectorized folded-decoder membership == reference loop, and == ML
# --------------------------------------------------------------------------- #
def _loop_reference_mask(rx, con, ctx):
    """A reference per-symbol Python-loop membership test, recomputed here so the
    vectorized np.isin path is checked against an independent implementation."""
    ab = t.nearest_lattice_point(rx, con.scale)
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    n = a * a + a * b + b * b
    sec = t.phase_pair_sector_array(ab)
    r_sq = ctx.r_sq
    r4 = r_sq * r_sq
    if ctx.fold_inversion:
        is_inner = n <= r_sq
        div = (n > 0) & (r4 % np.where(n > 0, n, 1) == 0)
        fold_n = np.where(is_inner, n, np.where(div, r4 // np.where(n > 0, n, 1), -1))
    else:
        fold_n = n
    stored = set(ctx.stored_shells)
    return np.array([(int(fn) in stored) for fn in fold_n], dtype=bool) & (sec >= 0)


@pytest.mark.parametrize("fold", [False, True])
def test_folded_mask_matches_loop_reference(fold):
    con = t.build_radial_dual_constellation(12, 60)
    ctx = t.make_folded_decode_context(con, fold_inversion=fold)
    rng = np.random.default_rng(3)
    tx = rng.integers(0, con.size, 40_000)
    rx = t.awgn(con.points_unit[tx], 12.0, con.bits_per_symbol, rng)
    _idx, fast_mask = t.decode_hex_folded(rx, ctx)
    assert np.array_equal(fast_mask, _loop_reference_mask(rx, con, ctx))


@pytest.mark.parametrize("fold", [False, True])
def test_folded_decode_equals_ml(fold):
    con = t.build_radial_dual_constellation(12, 60)
    ctx = t.make_folded_decode_context(con, fold_inversion=fold)
    rng = np.random.default_rng(11)
    tx = rng.integers(0, con.size, 60_000)
    rx = t.awgn(con.points_unit[tx], 10.0, con.bits_per_symbol, rng)
    idx, _fast = t.decode_hex_folded(rx, ctx)
    assert np.array_equal(idx, t.decode_ml(rx, con.points_unit))


def test_stored_arr_matches_stored_shells():
    con = t.build_radial_dual_constellation(12, 60)
    for fold in (False, True):
        ctx = t.make_folded_decode_context(con, fold_inversion=fold)
        assert isinstance(ctx.stored_arr, np.ndarray)
        assert ctx.stored_arr.dtype == np.int64
        assert np.array_equal(ctx.stored_arr, np.asarray(ctx.stored_shells,
                                                         dtype=np.int64))

# --------------------------------------------------------------------------- #
# Study 5 offset-sweep sanity (per-information-bit Eb/N0)
# --------------------------------------------------------------------------- #
def test_sim05_offset_sweep_sanity():
    rows = sim05.run_offset_sweep([0.0, 60.0], 5000, 10.0, 42)
    by = {int(r["dtheta_deg"]): r for r in rows}
    # A 60 deg offset slips the coherent sector (SER -> ~1) but the differential
    # scheme returns to the noise floor.
    assert by[60]["coherent_ser"] > 0.8
    assert by[60]["differential_ser"] < 0.05
    assert by[0]["coherent_ser"] < 0.05
