"""
test_packing_gain.py - Packing-Gain Readout Tests (Claim C3, Simulation 02)

Tests for the packing-gain readout helpers in Simulation 02 (claim C3).

These exercise the shipped simulation helpers directly (importing the simulation
module, whose ``main()`` is guarded), so the tests validate the actual code that
produces the paper's C3 headline numbers -- the dB packing gain with its
confidence band, and the impulsive error floor -- not a re-implementation.

Each row dict mirrors what ``run_channel`` emits: ``ebn0_db`` plus per-point SER
medians and exact Clopper-Pearson bounds (``hex_ser``/``hex_ser_lo``/``hex_ser_hi``
and the ``sq_`` counterparts).

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Version: 1.0.0
Date: June 24, 2026
"""
import numpy as np
import pytest

import simulation_02_hex_vs_square_ber_packing_gain as s2
import tqf_hex_signal as t


def _row(ebn0, hex_ser, sq_ser, ci=1.0):
    """A synthetic SER row. ``ci`` widens the symmetric (multiplicative) CI band:
    ci=1.0 -> degenerate (lo=hi=ser); ci>1 -> [ser/ci, ser*ci]."""
    return {
        "ebn0_db": ebn0,
        "hex_ser": hex_ser, "hex_ser_lo": hex_ser / ci, "hex_ser_hi": hex_ser * ci,
        "sq_ser": sq_ser, "sq_ser_lo": sq_ser / ci, "sq_ser_hi": sq_ser * ci,
    }


# --------------------------------------------------------------------------- #
# log-linear SER interpolation (the basis of every dB-gain readout)
# --------------------------------------------------------------------------- #
def test_interp_recovers_known_log_linear_crossing():
    # log10(SER) is linear in Eb/N0 here: SER=1e-2 at 10 dB, 1e-4 at 12 dB, so the
    # 1e-3 crossing must land exactly at the midpoint, 11.0 dB.
    rows = [_row(10.0, 1e-2, 1e-2), _row(12.0, 1e-4, 1e-4)]
    eb = s2._interp_ebn0_at_ser(rows, "hex_ser", 1e-3)
    assert eb == pytest.approx(11.0, abs=1e-9)


def test_interp_returns_none_when_target_not_bracketed():
    # The curve never reaches the target (everything well above 1e-3).
    rows = [_row(0.0, 0.4, 0.4), _row(4.0, 0.2, 0.2), _row(8.0, 0.1, 0.1)]
    assert s2._interp_ebn0_at_ser(rows, "hex_ser", 1e-3) is None


# --------------------------------------------------------------------------- #
# C3 dB packing gain with confidence band
# --------------------------------------------------------------------------- #
def test_gain_ci_band_brackets_the_point_estimate():
    # Hex reaches the target at a lower Eb/N0 than square -> positive gain. With a
    # non-degenerate CI, the band must contain the point estimate.
    ebn0 = [6.0, 8.0, 10.0, 12.0, 14.0]
    hex_ser = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]      # crosses 1e-2 at 10 dB
    sq_ser = [1.2e-1, 4e-2, 1.3e-2, 4e-3, 1.3e-3]  # crosses 1e-2 just after 10 dB
    rows = [_row(e, h, s, ci=1.4) for e, h, s in zip(ebn0, hex_ser, sq_ser)]
    g = s2._gain_with_ci(rows, 1e-2)
    assert g["bracketed"] is True
    assert g["gain_db"] > 0
    assert g["gain_lo_db"] <= g["gain_db"] <= g["gain_hi_db"]


def test_gain_ci_resolved_true_when_bands_disjoint():
    # Tight CIs around a clear hex advantage -> the conservative gain stays
    # positive, so ci_resolved must be True.
    ebn0 = [8.0, 10.0, 12.0]
    rows = [_row(e, h, s, ci=1.02) for e, h, s in
            zip(ebn0, [3e-2, 1e-2, 3e-3], [6e-2, 2e-2, 6e-3])]
    g = s2._gain_with_ci(rows, 1e-2)
    assert g["bracketed"] is True
    assert g["gain_lo_db"] is not None and g["gain_lo_db"] > 0
    assert g["ci_resolved"] is True


def test_gain_ci_resolved_false_when_bands_overlap():
    # A tiny hex edge swamped by wide CIs -> conservative gain goes negative, so
    # the difference is NOT statistically resolved.
    ebn0 = [8.0, 10.0, 12.0]
    rows = [_row(e, h, s, ci=3.0) for e, h, s in
            zip(ebn0, [3e-2, 1e-2, 3e-3], [3.2e-2, 1.05e-2, 3.2e-3])]
    g = s2._gain_with_ci(rows, 1e-2)
    assert g["bracketed"] is True
    assert g["ci_resolved"] is False


def test_gain_not_bracketed_when_floor_sits_above_target():
    # Impulsive-style floor: SER never drops below ~0.1, so a 1e-2 target is
    # unreachable -> not bracketed (reported as robustness parity, not a gain).
    ebn0 = [0.0, 8.0, 16.0, 24.0]
    rows = [_row(e, 0.1, 0.1) for e in ebn0]
    g = s2._gain_with_ci(rows, 1e-2)
    assert g["bracketed"] is False
    assert g["gain_db"] is None


# --------------------------------------------------------------------------- #
# impulsive error floor  ~  p * (1 - 1/M)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m", [16, 64, 256])
def test_impulsive_floor_matches_formula_and_recovers_p(m):
    p = 0.1
    floor = p * (1.0 - 1.0 / m)
    # A flat high-SNR tail sitting exactly on the theoretical floor.
    rows = [_row(e, floor, floor) for e in (40.0, 44.0, 48.0)]
    info = s2._impulsive_floor(rows, m, p)
    assert info["theoretical_floor"] == pytest.approx(floor, abs=1e-12)
    assert info["measured_floor_hex"] == pytest.approx(floor, abs=1e-12)
    assert info["implied_p"] == pytest.approx(p, abs=1e-9)


# --------------------------------------------------------------------------- #
# paired per-point significance: exact McNemar p-value (library helper)
# --------------------------------------------------------------------------- #
def test_mcnemar_no_discordant_pairs_is_one():
    # No trial distinguishes the two methods -> nothing to resolve.
    assert t.mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_is_symmetric_in_its_arguments():
    # The p-value depends only on the discordant counts, not which is "first".
    assert t.mcnemar_pvalue(7, 2) == t.mcnemar_pvalue(2, 7)


def test_mcnemar_strong_imbalance_is_significant():
    # 20 vs 2 discordant: exact two-sided p = 2 * P(Bin(22,1/2) <= 2)
    #   = 2 * (1 + 22 + 231) / 2**22 = 508 / 4194304.
    assert t.mcnemar_pvalue(20, 2) == pytest.approx(508 / 4194304, rel=1e-6)
    assert t.mcnemar_pvalue(20, 2) < 0.001


def test_mcnemar_balanced_is_not_significant():
    # 6 vs 4 discordant: exact two-sided p = 2 * P(Bin(10,1/2) <= 4)
    #   = 2 * (1 + 10 + 45 + 120 + 210) / 1024 = 772 / 1024.
    assert t.mcnemar_pvalue(6, 4) == pytest.approx(772 / 1024, rel=1e-9)
    assert t.mcnemar_pvalue(6, 4) > 0.05


# --------------------------------------------------------------------------- #
# Holm-Bonferroni family-wise correction (library helper)
# --------------------------------------------------------------------------- #
def test_holm_step_down_matches_hand_computation():
    # p = [.01,.02,.03,.04], m = 4. Holm-adjusted = [.04,.06,.06,.06];
    # only the smallest survives at alpha = 0.05.
    reject, p_adj = t.holm_bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)
    assert np.allclose(p_adj, [0.04, 0.06, 0.06, 0.06])
    assert list(reject) == [True, False, False, False]


def test_holm_rejects_nothing_when_all_null():
    reject, _ = t.holm_bonferroni([0.5, 0.6, 0.7], alpha=0.05)
    assert not reject.any()


def test_holm_single_hypothesis_is_uncorrected():
    reject, p_adj = t.holm_bonferroni([0.04], alpha=0.05)
    assert p_adj[0] == pytest.approx(0.04)
    assert bool(reject[0]) is True


def test_holm_is_invariant_to_input_order():
    p = [0.001, 0.2, 0.013, 0.04, 0.5]
    rej0, adj0 = t.holm_bonferroni(p, alpha=0.05)
    perm = [2, 4, 0, 3, 1]
    inv = np.argsort(perm)
    rej1, adj1 = t.holm_bonferroni([p[i] for i in perm], alpha=0.05)
    assert np.array_equal(rej0, rej1[inv])
    assert np.allclose(adj0, adj1[inv])


# --------------------------------------------------------------------------- #
# integration: run_channel emits the paired columns (consistent + reproducible)
# --------------------------------------------------------------------------- #
def test_run_channel_emits_consistent_paired_columns():
    rng = np.random.default_rng(123)
    rows = s2.run_channel("awgn", 16, [4.0, 8.0], trials=4000, alpha=0.05,
                          p=0.1, amplitude=5.0, rng=rng)
    keys = ("hex_sym_err", "sq_sym_err", "n_hex_only_err", "n_sq_only_err",
            "mcnemar_p", "mcnemar_p_holm", "ser_sig_holm", "trials")
    for r in rows:
        assert all(k in r for k in keys)
        assert 0.0 <= r["mcnemar_p"] <= 1.0
        assert 0.0 <= r["mcnemar_p_holm"] <= 1.0
        assert r["mcnemar_p_holm"] >= r["mcnemar_p"] - 1e-12     # Holm never lowers p
        assert r["ser_sig_holm"] in ("hex<sq", "hex>sq", "tie")
        assert r["n_hex_only_err"] + r["n_sq_only_err"] <= r["trials"]
        assert r["trials"] == 4000


def test_run_channel_is_reproducible_under_fixed_seed():
    # Determinism guard: the CRN rewind plus the new significance layer must stay
    # a pure function of the seed (so re-runs and the paper's numbers reproduce).
    def go():
        return s2.run_channel("rayleigh", 16, [8.0], trials=3000, alpha=0.05,
                              p=0.1, amplitude=5.0,
                              rng=np.random.default_rng(99))
    a, b = go(), go()
    assert a[0]["hex_sym_err"] == b[0]["hex_sym_err"]
    assert a[0]["sq_sym_err"] == b[0]["sq_sym_err"]
    assert a[0]["mcnemar_p"] == b[0]["mcnemar_p"]
