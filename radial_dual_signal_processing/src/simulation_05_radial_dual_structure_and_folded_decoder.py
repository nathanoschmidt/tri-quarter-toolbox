"""
simulation_05_radial_dual_structure_and_folded_decoder.py - Study 5 (Episode III).

Backs claim C7: the radial dual constellation (shell-complete, closed under the
order-6 rotation and circle inversion iota_r) admits an exact folded decoder that
stores only the inner + boundary shells and regenerates the outer shells from
their exact integer duals, while every decode decision remains a true Euclidean
nearest-point decision (the firewall: inversion regenerates stored labels, never
a distance). This study merges the structural verification and the folded decoder
and measures the fold across the family ladder.

What it measures, per inversion-paired family member (M = 42, 48, 54, 60)
------------------------------------------------------------------------
Structure (exact, integer):
  * phase-pair uniformity (equal per-sector occupancy on every complete shell);
  * inversion pairing (the exact iota_r point permutation is an involution that
    pairs shell N with shell r^4/N);
  * the self-dual boundary shell (N == r^2), when present.

Fold (exact, Burnside cross-checked against tqf_admissibility):
  * the Z6 label fold (rotation only): store one shell representative, regenerate
    six sector points -> 6x;
  * the Z6 x Z2 label fold (rotation + inversion): store inner + boundary shells,
    regenerate outer shells by exact dual -> the measured factor, whose ceiling
    is 12 and which falls short exactly on self-dual members (their boundary
    shell is inversion-fixed and does not fold): 21/2 = 10.5 at M = 42, 54/5 =
    10.8 at M = 54, 12 at the non-self-dual M = 48 and 60;
  * confirmation (Burnside) that D6 x Z2 = Z6 x Z2 here -- reflections add nothing
    to the label fold;
  * absolute stored table size (stored shells vs. full shells) and stored bytes.

Decode (exact + timed):
  * decode_hex_folded is verified bitwise-identical to exhaustive ML on a noisy
    stream at every M;
  * folded vs. unfolded vs. exhaustive-ML throughput, reported honestly: the fold
    is a storage/exactness property, and folded decode is typically a small
    wall-clock loss (the outer-shell dual regeneration costs more than it saves at
    these orders), which the table shows rather than hides.

Outputs
-------
  sim05_structure.csv     per-member structure flags, folds, stored sizes, bytes
  sim05_fold_audit.csv    exact Z6 / Z6xZ2 / D6xZ2 folds (fractions) per member
  sim05_throughput.csv    folded / unfolded / ML ns-per-symbol per member
  sim05_provenance.json
  (CSV column names keep the historical c6/d6 spelling for continuity with
  committed results; the paper writes the rotation group as Z6.)

Reproduce: python simulation_05_radial_dual_structure_and_folded_decoder.py

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np

import tqf_hex_signal as h
import tqf_admissibility as adm

SEED = 42
# (r_sq, max_norm_sq) that build the inversion-PAIRED ladder members M=42,48,54,60.
LADDER = [(48, 192), (18, 108), (12, 144), (24, 192)]
TIMING_SYMBOLS = 200_000
TIMING_REPEATS = 5
EBN0_TIMING_DB = 15.0
BYTES_PER_SHELL_ENTRY = 8      # one int64 shell norm per stored shell


def _label_fold_exact(con: h.Constellation, include_reflections: bool) -> Fraction:
    """Exact label fold for this constellation via the admissibility Burnside
    routine over its actual shell set."""
    return adm.burnside_label_fold(list(con.shell_norms), int(con.inversion_r_sq),
                                   include_reflections=include_reflections)


def _median_ns(fn, n_symbols: int, repeats: int) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) / n_symbols * 1e9)
    return float(np.median(ts))


def _analyze(r_sq: int, max_norm: int, rng: np.random.Generator
             ) -> Tuple[dict, dict, dict]:
    con = h.build_radial_dual_constellation(r_sq=r_sq, max_norm_sq=max_norm)
    M = con.size
    assert con.inversion_paired, f"member r_sq={r_sq} is not inversion-paired"
    assert con.phase_pair_uniform, f"member r_sq={r_sq} is not phase-pair uniform"

    # Folded (rotation + inversion) and rotation-only contexts.
    ctx_fold = h.make_folded_decode_context(con, fold_inversion=True)
    ctx_rot = h.make_folded_decode_context(con, fold_inversion=False)

    stored_fold = ctx_fold.stored_table_size      # inner + boundary shells
    stored_rot = ctx_rot.stored_table_size        # all shells (rotation fold only)
    full_shells = len(con.shell_norms)

    # Exact label folds (Burnside).
    fold_c6 = _label_fold_exact(con, include_reflections=False)
    fold_d6 = _label_fold_exact(con, include_reflections=True)

    # Storage: full label table would hold M (shell, sector) entries; the
    # rotation fold stores one entry per complete shell; the inversion fold stores
    # only inner + boundary shells. Report both the stored-shell count and bytes.
    self_dual = int(con.inversion_r_sq) if int(con.inversion_r_sq) in con.shell_norms else None

    # Correctness: folded decode == exhaustive ML on a noisy stream.
    tx = rng.integers(0, M, size=TIMING_SYMBOLS)
    rx = h.awgn(con.points_unit[tx], EBN0_TIMING_DB, max(1.0, np.log2(M)), rng)
    idx_fold, fast_fold = h.decode_hex_folded(rx, ctx_fold)
    idx_ml = h.decode_ml(rx, con.points_unit)
    assert np.array_equal(idx_fold, idx_ml), f"folded != ML at M={M}"
    idx_rot, _ = h.decode_hex_folded(rx, ctx_rot)
    assert np.array_equal(idx_rot, idx_ml), f"rotation-fold != ML at M={M}"

    # Throughput.
    t_fold = _median_ns(lambda: h.decode_hex_folded(rx, ctx_fold),
                        TIMING_SYMBOLS, TIMING_REPEATS)
    t_rot = _median_ns(lambda: h.decode_hex_folded(rx, ctx_rot),
                       TIMING_SYMBOLS, TIMING_REPEATS)
    t_ml = _median_ns(lambda: h.decode_ml(rx, con.points_unit),
                      TIMING_SYMBOLS, TIMING_REPEATS)

    structure = {
        "M": M, "r_sq": r_sq, "max_norm_sq": max_norm,
        "num_shells": full_shells,
        "self_dual_shell": self_dual if self_dual is not None else -1,
        "phase_pair_uniform": int(con.phase_pair_uniform),
        "inversion_paired": int(con.inversion_paired),
        "fundamental_domain_size": con.fundamental_domain_size,
        "stored_shells_rotfold": stored_rot,
        "stored_shells_invfold": stored_fold,
        "full_label_entries": M,
        "stored_bytes_rotfold": stored_rot * BYTES_PER_SHELL_ENTRY,
        "stored_bytes_invfold": stored_fold * BYTES_PER_SHELL_ENTRY,
        "full_label_bytes": M * BYTES_PER_SHELL_ENTRY,
        "label_fold_c6_z2": str(fold_c6),
        "label_fold_c6_z2_f": float(fold_c6),
    }
    audit = {
        "M": M, "r_sq": r_sq,
        "label_fold_c6": "6",                     # rotation-only fold is exactly 6
        "label_fold_c6_z2": str(fold_c6),
        "label_fold_d6_z2": str(fold_d6),
        "label_fold_c6_z2_f": float(fold_c6),
        "label_fold_d6_z2_f": float(fold_d6),
        "reflections_add_nothing": int(fold_c6 == fold_d6),
        "is_self_dual": int(self_dual is not None),
    }
    throughput = {
        "M": M, "r_sq": r_sq,
        "ns_folded_invfold": t_fold,
        "ns_folded_rotfold": t_rot,
        "ns_ml": t_ml,
        "fast_path_fraction": float(fast_fold.mean()),
        "folded_over_ml": t_fold / t_ml,
        "ebn0_db": EBN0_TIMING_DB,
    }
    return structure, audit, throughput


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_dir", type=str, default=".")
    args = ap.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    structures: List[dict] = []
    audits: List[dict] = []
    throughputs: List[dict] = []

    for r_sq, max_norm in LADDER:
        s, a, t = _analyze(r_sq, max_norm, rng)
        structures.append(s)
        audits.append(a)
        throughputs.append(t)

    # invariants: reflections never help the label fold; folds never exceed 12
    for a in audits:
        assert a["reflections_add_nothing"] == 1, f"D6 != Z6 label fold at M={a['M']}"
        assert a["label_fold_c6_z2_f"] <= 12.0 + 1e-9, f"label fold > 12 at M={a['M']}"
        # self-dual members must fall short of 12; non-self-dual must reach 12
        if a["is_self_dual"]:
            assert a["label_fold_c6_z2_f"] < 12.0, f"self-dual reached 12 at M={a['M']}"
        else:
            assert abs(a["label_fold_c6_z2_f"] - 12.0) < 1e-9, \
                f"non-self-dual missed 12 at M={a['M']}"

    def _write(name: str, rows: List[dict]) -> None:
        path = os.path.join(args.results_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write("sim05_structure.csv", structures)
    _write("sim05_fold_audit.csv", audits)
    _write("sim05_throughput.csv", throughputs)

    provenance = {
        "study": 5, "seed": SEED,
        "python": platform.python_version(), "numpy": np.__version__,
        "tqf_hex_signal_version": h.__version__,
        "ladder_r_sq_max_norm": LADDER,
        "timing_symbols": TIMING_SYMBOLS, "timing_repeats": TIMING_REPEATS,
        "ebn0_timing_db": EBN0_TIMING_DB,
        "bytes_per_shell_entry": BYTES_PER_SHELL_ENTRY,
    }
    with open(os.path.join(args.results_dir, "sim05_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("Study 5 complete.")
    for s, a, t in zip(structures, audits, throughputs):
        print(f"  M={s['M']:>3}  shells={s['num_shells']} "
              f"stored(rot->inv)={s['stored_shells_rotfold']}->{s['stored_shells_invfold']}  "
              f"labelfold Z6xZ2={a['label_fold_c6_z2']:>5} "
              f"(self_dual={a['is_self_dual']})  "
              f"folded/ML={t['folded_over_ml']:.2f}x")


if __name__ == "__main__":
    main()
