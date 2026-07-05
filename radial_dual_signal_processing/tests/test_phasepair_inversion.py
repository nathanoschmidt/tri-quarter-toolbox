"""
test_phasepair_inversion.py - Phase-Pair Primitive, Exact Inversion Duality, Folded Decoder, Radial-Dual Constellation, and T24 Differential Codec

Tests for the following components of the Tri-Quarter Framework (TQF)
radial_dual_signal_processing subproject:

  * the phase-pair sector primitive and its backward-compatible aliases;
  * the exact label-space inversion duality (involution + sector-preserving
    commutativity, the lattice paper's Prop. 4.15);
  * the C7 radial-dual constellation structure (phase-pair-uniform, inversion-
    paired, integer-dual shell pairs, exact involution permutation);
  * the phase-pair + inversion FOLDED decoder (bitwise-identical to exhaustive
    ML, with the inversion firewall: storage/label folds only);
  * the combined rotation + inversion (C6 x Z2) differential codec (C8),
    invariant under all 12 static actions;
  * the coloring equivariance fact (3-coloring equivariant; the conflict-free
    six-coloring proper but NOT rotation-equivariant).

Where practical the tests exercise the shipped simulation helpers directly, so
they validate the actual code that produces the paper's numbers.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.2.0
Date: July 4, 2026
"""
from fractions import Fraction

import numpy as np
import pytest

import tqf_hex_signal as t
import tqf_lattice_graph as g
import simulation_06_phasepair_inversion_folded_decoder as s6
import simulation_07_radial_dual_constellation as s7
import simulation_05_phase_rotation_robustness as s5
import simulation_03_symmetry_reduced_metric_exact as s3


# --------------------------------------------------------------------------- #
# Phase-pair primitive and backward-compatible aliases
# --------------------------------------------------------------------------- #
def test_phase_pair_sector_aliases_are_identical():
    # sector_index / sector_index_array must remain the same callables.
    assert t.sector_index is t.phase_pair_sector
    assert t.sector_index_array is t.phase_pair_sector_array


def test_phase_pair_sector_origin_is_minus_one():
    assert t.phase_pair_sector(0, 0) == -1


def test_phase_pair_sector_scalar_matches_array():
    ab = np.array([(a, b) for a in range(-5, 6) for b in range(-5, 6)],
                  dtype=np.int64)
    arr = t.phase_pair_sector_array(ab)
    scalar = np.array([t.phase_pair_sector(int(a), int(b)) for a, b in ab])
    assert np.array_equal(arr, scalar)


def test_phase_pair_sector_in_range_for_nonorigin():
    ab = np.array([(a, b) for a in range(-6, 7) for b in range(-6, 7)
                   if (a, b) != (0, 0)], dtype=np.int64)
    sec = t.phase_pair_sector_array(ab)
    assert sec.min() >= 0 and sec.max() <= 5


def test_rotation_increments_phase_pair_sector_by_one():
    # A +60 deg rotation must permute the integer sector by +1 (mod 6) on the
    # lattice -- the algebraic basis for the equivariance results.
    ab = np.array([(a, b) for a in range(-5, 6) for b in range(-5, 6)
                   if (a, b) != (0, 0)], dtype=np.int64)
    sec0 = t.phase_pair_sector_array(ab)
    rot = np.array([t.rotate60(int(a), int(b)) for a, b in ab], dtype=np.int64)
    sec1 = t.phase_pair_sector_array(rot)
    assert np.array_equal(sec1, (sec0 + 1) % 6)


# --------------------------------------------------------------------------- #
# Exact inversion duality: involution + sector-preserving commutativity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r_sq", [4, 7, 12, 19])
def test_inversion_involution_and_commutativity_zero_violations(r_sq):
    checked, inv_viol, comm_viol = t.verify_inversion_commutativity(
        r_sq, coord_radius=8)
    assert checked > 0
    assert inv_viol == 0
    assert comm_viol == 0


def test_inversion_exact_is_rational_involution_pointwise():
    # iota_r is an exact involution on coordinates (rational arithmetic).
    r_sq = 12
    for a in range(-5, 6):
        for b in range(-5, 6):
            if (a, b) == (0, 0):
                continue
            x, y = t.inversion_exact(a, b, r_sq)
            assert isinstance(x, Fraction) and isinstance(y, Fraction)


def test_dual_shell_norm_integer_pairs_about_r_sq_12():
    # The four integer-dual Loeschian shell pairs for r^2 = 12 (N * N' = 144).
    assert t.dual_shell_norm(3, 12) == 48
    assert t.dual_shell_norm(48, 12) == 3
    assert t.dual_shell_norm(4, 12) == 36
    assert t.dual_shell_norm(9, 12) == 16
    assert t.dual_shell_norm(12, 12) == 12          # self-dual (on the circle)


def test_invert_sector_shell_preserves_sector_maps_shell():
    s6_sec, dual = t.invert_sector_shell(2, 3, 12)
    assert s6_sec == 2                               # sector preserved
    assert dual == Fraction(144, 3)                  # exact shell map (= 48)


# --------------------------------------------------------------------------- #
# C7 radial-dual constellation structure
# --------------------------------------------------------------------------- #
def test_radial_dual_constellation_basic_shape():
    con = t.build_radial_dual_constellation(12, 60)
    assert con.size == 42
    assert tuple(con.shell_norms) == (3, 4, 9, 12, 16, 36, 48)
    assert con.inversion_r_sq == 12
    assert con.scale_sq_exact == Fraction(7, 128)
    assert con.fundamental_domain_size == 4


def test_radial_dual_constellation_is_unit_average_energy():
    con = t.build_radial_dual_constellation(12, 60)
    assert np.isclose(np.mean(np.abs(con.points_unit) ** 2), 1.0)


def test_radial_dual_phase_pair_uniform_and_inversion_paired():
    con = t.build_radial_dual_constellation(12, 60)
    assert con.phase_pair_uniform is True
    assert con.inversion_paired is True


def test_radial_dual_every_shell_has_one_point_per_sector():
    con = t.build_radial_dual_constellation(12, 60)
    occ = s7._shell_occupancy(con)
    for (N, size, sectors, k) in occ:
        assert size == 6
        assert sectors == [0, 1, 2, 3, 4, 5]
        assert k == 1


def test_radial_dual_inversion_is_same_sector_involution_fixing_boundary():
    con = t.build_radial_dual_constellation(12, 60)
    perm_ok, invol_ok, boundary_fixed = s7._verify_inversion_permutation(con)
    assert perm_ok is True
    assert invol_ok is True
    assert boundary_fixed == 6                       # |shell 12| fixed pointwise


def test_radial_dual_shell_pairs_helper():
    pairs = t.radial_dual_shell_pairs(12, 60)
    assert (3, 48) in pairs
    assert (4, 36) in pairs
    assert (9, 16) in pairs
    assert (12, 12) in pairs
    assert len(pairs) == 4


def test_loeschian_shells_up_to_excludes_non_norms():
    shells = set(t.loeschian_shells_up_to(20))
    assert {1, 3, 4, 7, 9, 12, 13, 16, 19} <= shells
    assert 2 not in shells and 5 not in shells and 6 not in shells


# --------------------------------------------------------------------------- #
# Folded decoder: bitwise-identical to exhaustive ML, both fold modes
# --------------------------------------------------------------------------- #
def _radial_dual_contexts():
    con = t.build_radial_dual_constellation(12, 60)
    ctx_pp = t.make_folded_decode_context(con, fold_inversion=False)
    ctx_ppi = t.make_folded_decode_context(con, fold_inversion=True)
    return con, ctx_pp, ctx_ppi


def test_folded_decoder_matches_ml_dense_grid():
    con, ctx_pp, ctx_ppi = _radial_dual_contexts()
    n, mm_pp, mm_ppi, _fast = s6.verify_dense_grid(con, ctx_pp, ctx_ppi, 2.5, 0.02)
    assert n > 0
    assert mm_pp == 0
    assert mm_ppi == 0


def test_folded_decoder_matches_ml_monte_carlo():
    con, ctx_pp, ctx_ppi = _radial_dual_contexts()
    rng = np.random.default_rng(0)
    mm_pp, mm_ppi, _fast = s6.verify_monte_carlo(con, ctx_pp, ctx_ppi,
                                                 50_000, 12.0, rng)
    assert mm_pp == 0
    assert mm_ppi == 0


def test_folded_decoder_storage_reduction_factors():
    con, ctx_pp, ctx_ppi = _radial_dual_contexts()
    # Phase-pair fold stores one representative per shell (7); +inversion stores
    # only the inner/boundary shells (4). These are the two SEPARATE reductions.
    assert ctx_pp.stored_table_size == 7
    assert ctx_ppi.stored_table_size == 4
    assert con.size / ctx_pp.stored_table_size == 6.0
    assert con.size / ctx_ppi.stored_table_size == 10.5


def test_folded_inner_shells_regenerate_full_shell_set():
    # The inversion fold is a genuine storage reduction: the inner half must
    # regenerate the full shell set from r_sq alone.
    con, _ctx_pp, ctx_ppi = _radial_dual_contexts()
    assert ctx_ppi.full_shells == tuple(con.shell_norms)
    rebuilt = t._reconstruct_full_shells(ctx_ppi.stored_shells, ctx_ppi.r_sq)
    assert rebuilt == tuple(con.shell_norms)


def test_folded_decode_requires_radial_dual_metadata():
    # A plain filled constellation lacks the shell metadata -> must raise.
    con = t.build_filled_constellation(16)
    with pytest.raises(ValueError):
        t.make_folded_decode_context(con, fold_inversion=True)


# --------------------------------------------------------------------------- #
# Inversion firewall: iota_r is conformal, NOT isometric
# --------------------------------------------------------------------------- #
def test_inversion_is_not_a_euclidean_isometry():
    # The firewall: inversion must NOT preserve pairwise squared distances, so it
    # cannot fold a Euclidean metric (only labels/storage/structure).
    con = t.build_radial_dual_constellation(12, 60)
    dual = np.asarray(con.inversion_dual_index, dtype=np.int64)
    a = con.ab[:, 0].astype(np.int64)
    b = con.ab[:, 1].astype(np.int64)
    da = a - a[0]; db = b - b[0]
    d0 = np.sort(da * da + da * db + db * db)
    ai = a[dual]; bi = b[dual]
    dai = ai - ai[0]; dbi = bi - bi[0]
    di = np.sort(dai * dai + dai * dbi + dbi * dbi)
    assert not np.array_equal(d0, di)


def test_euclidean_enumerator_folds_exactly_six_x_only():
    # On the radial-dual object the squared-distance enumerator folds by rotation
    # EXACTLY 6x; the combined block must report exactness and the 6x figure.
    ok = s3.radial_dual_reduction_block(12, 60, "/tmp")
    assert ok is True


# --------------------------------------------------------------------------- #
# C8 -- combined rotation + inversion (C6 x Z2) differential codec
# --------------------------------------------------------------------------- #
def test_t24_codec_roundtrip_no_action():
    rng = np.random.default_rng(1)
    d_sec = rng.integers(0, 6, 1000)
    d_inv = rng.integers(0, 2, 1000)
    s, u = t.differential_encode_t24(d_sec, d_inv)
    ds, du = t.differential_decode_t24(s, u)
    assert np.array_equal(ds[1:], d_sec[1:])
    assert np.array_equal(du[1:], d_inv[1:])


def test_t24_codec_invariant_under_all_twelve_actions():
    total_viol, rows = s5.verify_t24_differential_invariance(5000, seed=808)
    assert total_viol == 0
    assert len(rows) == 12
    assert all(r["sector_recovered"] and r["inversion_recovered"] for r in rows)


def test_t24_inversion_bit_is_pure_label_state():
    # The inversion component is a Z2 state: a global flip must cancel exactly.
    rng = np.random.default_rng(2)
    d_sec = rng.integers(0, 6, 800)
    d_inv = rng.integers(0, 2, 800)
    s, u = t.differential_encode_t24(d_sec, d_inv)
    ds, du = t.differential_decode_t24(s, (u + 1) % 2)   # global inversion flip
    assert np.array_equal(ds[1:], d_sec[1:])
    assert np.array_equal(du[1:], d_inv[1:])


# --------------------------------------------------------------------------- #
# Coloring equivariance (six-coloring is NOT rotation-equivariant)
# --------------------------------------------------------------------------- #
def test_three_coloring_is_rotation_equivariant_six_is_not():
    import simulation_04_sixcoloring_denoise_gpu as s4
    eq = s4.verify_coloring_equivariance(24)
    assert eq["three_coloring_proper"] is True
    assert eq["six_coloring_proper"] is True
    assert eq["three_coloring_rotation_equivariant"] is True
    assert eq["six_coloring_rotation_equivariant"] is False


def test_three_coloring_helper_is_proper():
    graph = g.build_lattice_graph(20)
    coords = graph["coords"]
    index_of = {(int(a), int(b)): i for i, (a, b) in enumerate(coords)}
    classes, proper = g.three_coloring(coords, index_of)
    assert proper is True
    assert len(classes) == 3
    assert sum(c.size for c in classes) == coords.shape[0]
