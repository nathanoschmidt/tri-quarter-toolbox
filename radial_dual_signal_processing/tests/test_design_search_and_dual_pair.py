"""
test_design_search_and_dual_pair.py - Symmetry-Reduced Design Search (C10) and Dual-Pair Transmission (C11) Tests

Topic-focused tests for the two Episode III studies (Studies 7 and 8) that build on the exact
admissibility machinery:

  * Study 7 (C10): an exhaustive constellation design search that canonicalizes
    candidate point sets under the dihedral isometry group D6 evaluates far fewer
    candidates than the unreduced search while finding the SAME exact optimum.
  * Study 8 (C11): transmitting the inversion pair (x, iota_r(x)) and decoding
    both legs with an exact integer consistency cross-check gives a rate-1/2 block
    code; inversion is not an isometry, so the joint spectrum is genuinely thinned
    relative to the (isometric) repetition and rotated-repetition baselines.

The helpers are imported from the shipped simulation modules, so these tests
validate the actual code that produces the paper's numbers.

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.3.0
Date: July 6, 2026
"""

import itertools

import numpy as np

import tqf_admissibility as adm
import simulation_07_design_search_symmetry as sim07
import simulation_08_dual_pair_transmission as sim08


# --------------------------------------------------------------------------- #
# C10 -- D6-canonical design-search reduction
# --------------------------------------------------------------------------- #
def test_objective_is_exact_min_distance_and_multiplicity():
    # Three collinear lattice points at unit steps: min squared distance 1,
    # realized by the two adjacent unordered pairs.
    assert sim07._objective(((0, 0), (1, 0), (2, 0))) == (1, 2)


def test_d6_canonicalization_reduces_and_preserves_optimum():
    pool = sim07._pool(3)                       # 12-point pool (shells 1 and 3)
    pool_set = set(pool)
    elems = adm._group_elements_geometric(include_reflections=True)
    subsets = list(itertools.combinations(pool, 3))

    canon = set()
    best_full = None
    for s in subsets:
        obj = sim07._objective(s)
        if best_full is None or sim07._better(obj, best_full):
            best_full = obj
        canon.add(sim07._d6_canonical(s, elems, pool_set))

    # The D6 fold strictly reduces the number of candidates to evaluate.
    assert len(canon) < len(subsets)

    # ... yet the exact optimum over canonical representatives is identical.
    best_canon = None
    for s in canon:
        obj = sim07._objective(s)
        if best_canon is None or sim07._better(obj, best_canon):
            best_canon = obj
    assert best_canon == best_full


def test_d6_canonical_is_invariant_under_the_group_action():
    # The pool is a union of complete shells (D6-closed), so a rotated subset
    # stays in the pool and must share the canonical representative.
    pool = sim07._pool(3)
    pool_set = set(pool)
    elems = adm._group_elements_geometric(include_reflections=True)
    subset = (pool[0], pool[4], pool[9])
    rot = elems[1]                              # the single-step order-6 rotation R_{pi/3}
    rotated = tuple(sorted(rot(a, b) for (a, b) in subset))
    assert (sim07._d6_canonical(subset, elems, pool_set)
            == sim07._d6_canonical(rotated, elems, pool_set))


# --------------------------------------------------------------------------- #
# C11 -- inversion-pair rate-1/2 block code
# --------------------------------------------------------------------------- #
def test_pair_codebook_is_inversion_paired_and_has_a_boundary():
    con, x, xinv, di, self_dual, shell_of = sim08._build_pair_codebook(12, 60)
    assert con.size == 42
    # xinv is exactly the dual-indexed primary constellation
    assert np.array_equal(xinv, x[di])
    # inversion is an involution on the labels
    assert np.array_equal(di[di], np.arange(con.size))
    # the self-dual (boundary) shell r_sq = 12 contributes six fixed points
    assert int(self_dual.sum()) == 6


def test_exact_product_distance_thins_the_spectrum_vs_isometric_baselines():
    con, *_ = sim08._build_pair_codebook(12, 60)
    pd = sim08._exact_product_distance(con, 12)
    # The isometry theorem: a rotated (isometric) second leg cannot thin the joint
    # spectrum, so it matches the plain repetition baseline exactly.
    assert pd["isometric_leg_equals_repetition"] == 1
    # Inversion is NOT an isometry, so the inversion-pair nearest-neighbor
    # multiplicity is strictly smaller than repetition's (a genuine reduction).
    assert pd["multiplicity_ratio_rep_over_pair"] > 1.0
    assert pd["self_dual_count"] == 6
    assert pd["d2min_pair_lattice"] > 0


def test_consistency_check_recovers_message_noiselessly_and_flags_corruption():
    con, x, xinv, di, self_dual, shell_of = sim08._build_pair_codebook(12, 60)
    tx = np.arange(con.size)

    # Noiseless: both legs land on their true points -> perfect recovery, no erasure.
    est, detected = sim08._per_symbol_with_check(x[tx], xinv[tx], con, x, xinv, di)
    assert np.array_equal(est, tx)
    assert not detected.any()

    # Corrupt the second leg with a non-dual point -> the exact integer consistency
    # cross-check must DETECT (erase) the mismatched pair.
    bad_leg = xinv[(tx + 1) % con.size]         # wrong dual for (almost) every message
    _est2, detected2 = sim08._per_symbol_with_check(x[tx], bad_leg, con, x, xinv, di)
    assert detected2.any()
