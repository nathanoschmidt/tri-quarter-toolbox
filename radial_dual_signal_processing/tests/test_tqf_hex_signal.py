"""
test_tqf_hex_signal.py - Core Hexagonal Signal Library Tests (Claim C1, C3 Preconditions)

Tests for tqf_hex_signal.py -- the core hexagonal signal library.

Covers the foundations of claims C1 (exact ML-equivalent demodulation) and C3's
fairness preconditions (unit-energy normalization), plus the exact integer/
rational primitives the framework relies on.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.1.0
Date: June 27, 2026
"""
import math

import numpy as np
import pytest

import tqf_hex_signal as t

M_VALUES = [16, 64, 256]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _brute_nearest(z, scale, w=5):
    """Ground-truth nearest A2 lattice point via a wide (2w+1)^2 search."""
    zl = np.asarray(z) / scale
    a0, b0 = t.complex_to_oblique(zl)
    na, nb = np.round(a0).astype(int), np.round(b0).astype(int)
    best = np.full(zl.shape, np.inf)
    ba, bb = na.copy(), nb.copy()
    for da in range(-w, w + 1):
        for db in range(-w, w + 1):
            ca, cb = na + da, nb + db
            cz = (ca + cb * 0.5) + 1j * (cb * (math.sqrt(3) / 2.0))
            d2 = np.abs(zl - cz) ** 2
            mask = d2 < best
            best = np.where(mask, d2, best)
            ba = np.where(mask, ca, ba)
            bb = np.where(mask, cb, bb)
    return np.stack([ba, bb], axis=1)


# --------------------------------------------------------------------------- #
# exact geometry primitives
# --------------------------------------------------------------------------- #
def test_eisenstein_norm_equals_squared_euclidean():
    for a in range(-6, 7):
        for b in range(-6, 7):
            z = t.lattice_to_complex(a, b)
            assert t.shell_norm_sq(a, b) == pytest.approx(abs(z) ** 2, abs=1e-9)


def test_oblique_complex_roundtrip():
    rng = np.random.default_rng(0)
    ab = rng.integers(-25, 25, size=(2000, 2))
    z = t.lattice_array_to_complex(ab)
    a_real, b_real = t.complex_to_oblique(z)
    assert np.allclose(a_real, ab[:, 0], atol=1e-9)
    assert np.allclose(b_real, ab[:, 1], atol=1e-9)


def test_sector_index_in_range_and_rotation_equivariant():
    # The order-6 rotation must permute the sector label by exactly +1 (mod 6) --
    # the integer-level statement underlying C6.
    for a in range(-8, 9):
        for b in range(-8, 9):
            if a == 0 and b == 0:
                continue
            s = t.sector_index(a, b)
            assert 0 <= s < 6
            ra, rb = t.rotate60(a, b)
            assert t.sector_index(ra, rb) == (s + 1) % 6


def test_sector_index_array_matches_scalar():
    ab = np.array([[a, b] for a in range(-6, 7) for b in range(-6, 7)
                   if (a, b) != (0, 0)], dtype=np.int64)
    arr = t.sector_index_array(ab)
    scal = np.array([t.sector_index(int(a), int(b)) for a, b in ab])
    assert np.array_equal(arr, scal)


def test_rotate60_is_order_six():
    # Applying the rotation six times returns the original point.
    for a, b in [(1, 0), (3, -2), (-4, 5), (7, 1)]:
        p = (a, b)
        for _ in range(6):
            p = t.rotate60(*p)
        assert p == (a, b)


# --------------------------------------------------------------------------- #
# constellations: energy normalization (C3 fairness precondition)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", M_VALUES)
def test_filled_constellation_unit_energy(m):
    con = t.build_filled_constellation(m)
    assert con.size == m
    assert con.bits_per_symbol == int(round(math.log2(m)))
    assert np.mean(np.abs(con.points_unit) ** 2) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("m", M_VALUES)
def test_square_qam_unit_energy(m):
    con = t.build_square_qam(m)
    assert con.size == m
    assert np.mean(np.abs(con.points_unit) ** 2) == pytest.approx(1.0, abs=1e-12)


def test_disk_constellation_six_fold_symmetric():
    # A disk constellation is closed under the order-6 rotation (used by C4/C6).
    con = t.build_disk_constellation(37)
    pts = {(int(a), int(b)) for a, b in con.ab}
    assert con.size % 6 == 0
    for a, b in con.ab:
        assert (int(a), int(b)) in pts
        ra, rb = t.rotate60(int(a), int(b))
        assert (ra, rb) in pts


# --------------------------------------------------------------------------- #
# C1: exact closed-form decode == exhaustive ML
# --------------------------------------------------------------------------- #
def test_nearest_lattice_point_matches_bruteforce():
    scale = 0.37
    rng = np.random.default_rng(1)
    n = 150_000
    z = (rng.uniform(-10, 10, n) + 1j * rng.uniform(-10, 10, n)) * scale
    # adversarial deep holes: centroids of lattice triangles ~ (a+1/3, b+1/3)
    base = rng.integers(-6, 6, size=(n, 2))
    ca = base[:, 0] + 1.0 / 3.0
    cb = base[:, 1] + 1.0 / 3.0
    zc = ((ca + cb * 0.5) + 1j * (cb * (math.sqrt(3) / 2.0))) * scale
    allz = np.concatenate([z, zc])
    fast = t.nearest_lattice_point(allz, scale)
    gt = _brute_nearest(allz, scale, w=5)
    assert np.array_equal(fast, gt)


@pytest.mark.parametrize("m", M_VALUES)
def test_decode_fast_equals_ml_on_noisy_stream(m):
    con = t.build_filled_constellation(m)
    ctx = t.make_hex_decode_context(con)
    rng = np.random.default_rng(2)
    tx = rng.integers(0, con.size, 40_000)
    for ebn0 in (0.0, 6.0, 12.0):
        rx = t.awgn(con.points_unit[tx], ebn0, con.bits_per_symbol, rng)
        fast, fast_mask = t.decode_hex_fast(rx, ctx)
        ml = t.decode_ml(rx, con.points_unit)
        assert np.array_equal(fast, ml)
        assert fast_mask.dtype == bool and fast_mask.shape == fast.shape


def test_decode_fast_equals_ml_on_dense_grid():
    con = t.build_filled_constellation(64)
    ctx = t.make_hex_decode_context(con)
    ext = float(np.max(np.abs(con.points_unit))) * 1.2
    g = np.linspace(-ext, ext, 220)
    X, Y = np.meshgrid(g, g)
    rx = (X + 1j * Y).ravel()
    fast, _mask = t.decode_hex_fast(rx, ctx)
    ml = t.decode_ml(rx, con.points_unit)
    assert np.array_equal(fast, ml)


def test_fast_path_fraction_increases_with_snr():
    con = t.build_filled_constellation(256)
    ctx = t.make_hex_decode_context(con)
    rng = np.random.default_rng(5)
    tx = rng.integers(0, con.size, 40_000)
    fracs = []
    for ebn0 in (0.0, 6.0, 12.0):
        rx = t.awgn(con.points_unit[tx], ebn0, con.bits_per_symbol, rng)
        _idx, mask = t.decode_hex_fast(rx, ctx)
        fracs.append(float(np.mean(mask)))
    assert fracs[0] <= fracs[1] <= fracs[2]
    assert fracs[-1] > 0.9


# --------------------------------------------------------------------------- #
# channels and statistics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("ebn0", [0.0, 6.0, 10.0])
def test_noise_sigma_mapping(ebn0, k):
    sig2 = t._noise_sigma_sq(ebn0, k)
    es_n0_lin = 10.0 ** ((ebn0 + 10.0 * math.log10(k)) / 10.0)
    assert sig2 == pytest.approx(1.0 / es_n0_lin, rel=1e-12)


def test_awgn_empirical_variance_matches_mapping():
    rng = np.random.default_rng(7)
    sym = np.ones(400_000, dtype=complex)
    ebn0, k = 6.0, 4
    rx = t.awgn(sym, ebn0, k, rng)
    noise = rx - sym
    emp = float(np.var(noise.real) + np.var(noise.imag))
    assert emp == pytest.approx(t._noise_sigma_sq(ebn0, k), rel=0.05)


def test_clopper_pearson_brackets_estimate():
    for errors, trials in [(0, 1000), (1, 1000), (50, 1000), (1000, 1000)]:
        lo, hi = t.clopper_pearson(errors, trials)
        p = errors / trials
        assert 0.0 <= lo <= p <= hi <= 1.0


def test_clopper_pearson_tightens_with_more_trials():
    _lo1, hi1 = t.clopper_pearson(10, 1_000)
    _lo2, hi2 = t.clopper_pearson(100, 10_000)  # same p=0.01, 10x trials
    width1 = hi1 - _lo1
    width2 = hi2 - _lo2
    assert width2 < width1


# --------------------------------------------------------------------------- #
# differential hexagonal coding (C6)
# --------------------------------------------------------------------------- #
def test_differential_roundtrip_zero_offset():
    rng = np.random.default_rng(4)
    d = rng.integers(0, 6, 5000)
    assert np.array_equal(t.differential_decode(t.differential_encode(d)), d)


@pytest.mark.parametrize("k", list(range(6)))
def test_differential_invariant_to_k_times_60deg(k):
    # A static sector shift by k (a multiple of pi/3) cancels in the decoder's
    # consecutive differences for every symbol after the reference.
    rng = np.random.default_rng(4)
    d = rng.integers(0, 6, 5000)
    enc = (t.differential_encode(d) + k) % 6
    dec = t.differential_decode(enc)
    assert np.array_equal(dec[1:], d[1:])


# --------------------------------------------------------------------------- #
# provenance / reproducibility metadata (Methods table; C5 CUDA certification)
# --------------------------------------------------------------------------- #
def test_collect_provenance_has_methods_table_fields():
    """The provenance dict must carry every field the paper's Methods table and
    the C5 GPU-certification depend on, with the right types.

    WHY:  every results CSV/JSON is paired with this record; a missing field
          would silently drop reproducibility information from the paper.
    HOW:  collect with no torch device (the default for the non-GPU studies).
    WHAT: required keys present; version fields are strings; ran_on_cuda is a
          bool; with no device passed, device_used is the literal 'n/a'.
    """
    prov = t.collect_provenance()
    required = {
        "tqf_version", "timestamp_utc", "platform", "python_version",
        "numpy_version", "scipy_version", "torch_installed",
        "torch_built_with_cuda", "cuda_is_available", "device_used",
        "ran_on_cuda",
    }
    assert required.issubset(prov.keys())
    assert isinstance(prov["python_version"], str)
    assert isinstance(prov["numpy_version"], str)
    assert isinstance(prov["ran_on_cuda"], bool)
    assert isinstance(prov["cuda_is_available"], bool)
    assert prov["device_used"] == "n/a"


def test_collect_provenance_ran_on_cuda_requires_a_cuda_device():
    """ran_on_cuda may only be True when a CUDA device is actually used.

    With no device passed it must be False -- this is the invariant that keeps
    Simulation 04's CSV ``ran_on_cuda`` column honest (it cannot be True unless
    a CUDA device was selected and available).
    """
    prov = t.collect_provenance(device=None)
    assert prov["ran_on_cuda"] is False
    # Internal consistency: ran_on_cuda implies cuda_is_available.
    assert (not prov["ran_on_cuda"]) or prov["cuda_is_available"]


def test_emit_provenance_writes_json_roundtrip(tmp_path):
    """emit_provenance writes a parseable ``<sim>_provenance.json`` and returns
    the same provenance dict it persisted (the durable Methods-table artifact)."""
    import json

    prov = t.emit_provenance(str(tmp_path), "sim_unit_test")
    out = tmp_path / "sim_unit_test_provenance.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["provenance"]["python_version"] == prov["python_version"]
    assert payload["provenance"]["ran_on_cuda"] == prov["ran_on_cuda"]
