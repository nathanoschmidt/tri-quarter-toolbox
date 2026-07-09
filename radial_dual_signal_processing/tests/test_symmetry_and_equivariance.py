"""Tests for the exact symmetry-reduced metric (C4) and decoder equivariance (C6).

These exercise the shipped simulation helpers directly (importing the simulation
modules, whose ``main()`` is guarded), so the tests validate the actual code that
produces the paper's numbers -- not a re-implementation.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 8, 2026
"""
import numpy as np
import pytest

import tqf_hex_signal as t
import simulation_03_symmetry_reduced_metric as s3
import simulation_04_phase_rotation_and_differential as s4

NORMS = [7, 19, 37, 61]


# --------------------------------------------------------------------------- #
# C4 -- exact symmetry-reduced performance metric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", NORMS)
def test_orbit_reduced_enumerator_equals_full(T):
    # The headline exactness check: the C6 orbit-folded squared-distance
    # enumerator is BITWISE identical to the full enumerator (exact integer
    # multiset equality on the Counter, not a floating-point tolerance).
    ab = t.build_disk_constellation(T).ab.astype(np.int64)
    full = s3._full_enumerator(ab)
    reps = s3._orbit_reps(ab, include_reflections=False)
    folded = s3._folded_enumerator(ab, reps, include_reflections=False)
    assert folded == full


@pytest.mark.parametrize("T", NORMS)
def test_d6_orbit_reduced_enumerator_equals_full(T):
    # The full dihedral D6 fold (rotations AND reflections, order 12) is also exact.
    ab = t.build_disk_constellation(T).ab.astype(np.int64)
    full = s3._full_enumerator(ab)
    reps = s3._orbit_reps(ab, include_reflections=True)
    folded = s3._folded_enumerator(ab, reps, include_reflections=True)
    assert folded == full


@pytest.mark.parametrize("T", NORMS)
def test_orbit_count_is_one_sixth(T):
    # A disk constellation is C6-closed with the origin excluded, so every orbit
    # has size 6 and the representative count is exactly M / 6.
    ab = t.build_disk_constellation(T).ab.astype(np.int64)
    reps = s3._orbit_reps(ab, include_reflections=False)
    assert len(reps) * 6 == ab.shape[0]


@pytest.mark.parametrize("T", NORMS)
def test_exact_six_times_operation_reduction(T):
    ab = t.build_disk_constellation(T).ab.astype(np.int64)
    m = ab.shape[0]
    reps = s3._orbit_reps(ab, include_reflections=False)
    full_evals = m * (m - 1)
    orbit_evals = len(reps) * (m - 1)
    assert full_evals == 6 * orbit_evals


def test_disk_min_distance_squared_is_one():
    ab = t.build_disk_constellation(37).ab.astype(np.int64)
    full = s3._full_enumerator(ab)
    assert min(full.keys()) == 1  # nearest neighbors at unit squared lattice distance


# --------------------------------------------------------------------------- #
# C6 -- decoder equivariance on a real 2-D hex constellation + differential
# --------------------------------------------------------------------------- #
def test_decoder_equivariance_on_real_hex_constellation():
    # Rotating every constellation point by +60 deg must permute the decoded
    # sector index by exactly +1, with zero violations.
    npts, violations = s4.verify_equivariance_on_hex()
    assert npts > 0
    assert violations == 0


def test_differential_roundtrip_at_pi_over_3_multiples():
    assert s4.verify_differential_roundtrip() is True
