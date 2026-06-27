"""
test_symmetry_and_equivariance.py - Symmetry-Reduced Metric (C4) and Decoder Equivariance (C6) Tests

Tests for the exact symmetry-reduced metric (C4) and decoder equivariance (C6).

These exercise the shipped simulation helpers directly (importing the simulation
modules, whose ``main()`` is guarded), so the tests validate the actual code that
produces the paper's numbers -- not a re-implementation.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.1.0
Date: June 27, 2026
"""
import numpy as np
import pytest

import tqf_hex_signal as t
import simulation_03_symmetry_reduced_metric_exact as s3
import simulation_05_phase_rotation_robustness as s5

NORMS = [7, 19, 37, 61]


def _enum_size(ab):
    """Smallest bincount length that holds the maximum squared distance exactly."""
    a = ab[:, 0].astype(np.int64)
    b = ab[:, 1].astype(np.int64)
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    return int((da * da + da * db + db * db).max()) + 1


# --------------------------------------------------------------------------- #
# C4 -- exact symmetry-reduced performance metric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", NORMS)
def test_orbit_reduced_enumerator_equals_full(T):
    # The headline exactness check: orbit-reduced enumerator is BITWISE identical
    # to the full enumerator (integer ==, not a tolerance).
    ab = t.build_disk_constellation(T).ab
    size = _enum_size(ab)
    reps = np.array(s3._orbit_partition(ab), dtype=np.int64)
    full = s3._pair_enumerator_full(ab, size)
    orbit = s3._pair_enumerator_orbit(ab, reps, size)
    assert np.array_equal(full, orbit)


@pytest.mark.parametrize("T", NORMS)
def test_orbit_count_is_one_sixth(T):
    ab = t.build_disk_constellation(T).ab
    reps = s3._orbit_partition(ab)
    assert len(reps) * 6 == ab.shape[0]


@pytest.mark.parametrize("T", NORMS)
def test_exact_six_times_operation_reduction(T):
    ab = t.build_disk_constellation(T).ab
    m = ab.shape[0]
    reps = s3._orbit_partition(ab)
    full_evals = m * (m - 1)
    orbit_evals = len(reps) * (m - 1)
    assert full_evals == 6 * orbit_evals


def test_disk_min_distance_squared_is_one():
    ab = t.build_disk_constellation(37).ab
    enum = s3._pair_enumerator_full(ab, _enum_size(ab))
    nz = np.nonzero(enum)[0]
    nz = nz[nz > 0]
    assert nz.min() == 1  # nearest neighbours at unit squared lattice distance


# --------------------------------------------------------------------------- #
# C6 -- decoder equivariance on a real 2D hex constellation + differential
# --------------------------------------------------------------------------- #
def test_decoder_equivariance_on_real_hex_constellation():
    # Rotating every constellation point by +60 deg must permute the decoded
    # sector index by exactly +1, with zero violations.
    npts, violations = s5.verify_equivariance_on_hex()
    assert npts > 0
    assert violations == 0


def test_differential_roundtrip_at_pi_over_3_multiples():
    assert s5.verify_differential_roundtrip() is True
