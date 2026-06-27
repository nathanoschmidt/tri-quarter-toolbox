# TQF Radial Dual Signal Processing: Everything You Ever Wanted to Know (And Some Things You Didn't)

*A Q&A guide for people who know what a constellation diagram is but have not yet started seeing hexagons in their sleep.*

*This document is primarily machine-generated from the codebase and is an experiment in itself.*

---

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.0<br>
**Last Updated:** June 27, 2026<br>

---

**A note on honesty before we begin.** This subproject is unusually disciplined about
what it does and does not claim. The closed-form decoder *is* mathematically equal to
optimal (maximum-likelihood) decoding, which means it does **not** beat ML on error
rate — it cannot, because it *is* ML. So every win here lives in one of two buckets:
**cost** (you compute the same answer with fewer operations, less storage, or more
parallelism) or **geometry** (the constellation's shape and symmetry give a packing
or robustness advantage). If you keep those two buckets in mind, the whole document
will make sense. We will never quietly slide a cost win into the geometry column.

---

## Part 1: The Big Picture

---

**Q: What is this subproject, in one breath?**

A: It takes the Tri-Quarter Framework (TQF) — a hexagonal-lattice mathematical
machinery — and applies it to **digital communications**. Where a sibling case study
applied TQF to 1D BPSK (two points on a line), this one moves to **2D hexagonal
signal constellations**: the points you transmit over a channel are arranged on a
hexagonal lattice instead of the usual square grid. The framework's order-6 rotational
symmetry then does real, measurable work — in decoding cost, in packing efficiency,
and in robustness to phase and amplitude ambiguities.

The unifying discipline: every claim in the companion paper is backed by a script here
that prints a clean results table, *and* by an automated test that pins the underlying
mathematical invariant. Eight claims, labeled C1 through C8.

---

**Q: What problem does it actually solve?**

A: When a receiver gets a noisy signal point, it has to decide which constellation
point you most likely sent. That's **demodulation/decoding**. Two things matter:
getting the *right* answer (optimal = maximum-likelihood), and getting it *cheaply*
(fast, low-storage, parallelizable). This subproject shows that on a hexagonal
lattice you can do optimal decoding with a closed-form, mostly-constant-time decoder,
get a modest packing advantage over square QAM "for free" from the geometry, and
exploit the lattice's symmetry to slash computation and storage — all with exact
integer/rational arithmetic so the exactness claims have *no* floating-point wiggle room.

---

**Q: Why hexagons? What's wrong with the square grid everyone already uses?**

A: Nothing is *wrong* with square QAM — it's everywhere and it works. But the
hexagonal lattice (formally the **A₂ lattice**, described with Eisenstein integers) is
the densest packing of points in 2D. For a fixed minimum distance between points,
hexagons fit more of them into a given average energy budget — or equivalently, at a
matched number of points and matched energy, they spread a little farther apart. That
spacing is the "packing gain" (C3). Hexagons also have 6-fold rotational symmetry,
which is the lever every cost-and-robustness claim in this project pulls on.

---

**Q: What does "radial dual" mean? That's in the project name.**

A: It refers to a duality created by **circle inversion** — a transformation that
swaps the inside and outside of a circle of radius r. Applied to the lattice, it pairs
each "shell" (set of points at a given squared distance from the origin) with a **dual
shell** on the other side of the inversion circle. The headline object (C7) is a
constellation that is *closed* under this inversion: it looks the same after you turn
it inside-out about r²=12. "Radial" because shells are indexed by radius; "dual"
because inversion pairs them. This is the part of the framework that goes beyond
rotation alone.

---

**Q: How does this relate to the other TQF papers?**

A: It is the 2D sequel. The lineage:
1. The foundational TQF paper (complex coordinates with topological and reflective
   duality across circles of any radius).
2. The BPSK case study (TQF applied to the simplest 1D modulation).
3. The radial-dual triangular lattice graph companion (the geometric/graph object).
4. **This subproject** — carrying all of that into 2D hexagonal signal processing.

You don't need to have read the others; this document is self-contained. But the
inversion duality used in C7 cites a proposition (Prop. 4.15) from the lattice-graph
companion rather than re-deriving it.

---

## Part 2: The Lattice and the Geometry

---

**Q: What are Eisenstein integers and the A₂ lattice?**

A: Eisenstein integers are complex numbers of the form `a + b·ω` where
`ω = e^(2πi/3)` and `a, b` are ordinary integers. They tile the plane in a hexagonal
pattern — every point has 6 nearest neighbors forming a perfect hexagon. That tiling
*is* the A₂ lattice, the densest lattice packing in two dimensions. In this codebase
the lattice points are the candidate constellation symbols, and because their
coordinates are integer combinations of a fixed basis, all the geometry (distances,
sectors, shells) can be done in **exact integer arithmetic**.

---

**Q: What are sectors, shells, and colours?**

A: Three different exact integer labels attached to each lattice point:

- **Sector** — which of the 6 angular wedges (each 60°) the point falls in. Rotating
  by 60° advances the sector by exactly 1. This is where the order-6 symmetry becomes
  a concrete, countable thing.
- **Shell** — the squared distance from the origin (an integer, e.g. 3, 4, 9, 12).
  Points on the same shell are related by rotation/reflection.
- **Colour** — a residue used to partition the lattice for parallel scheduling. The
  trihexagonal six-colouring is `2·((a−b) mod 3) + ((a+b) mod 2)`.

All three are integers, computed without ever touching a float — which is the whole
point of the "exactness" claims.

---

**Q: What is `phase_pair_sector` and why was it renamed?**

A: It's the integer primitive that maps a lattice point `(a, b)` to its phase-pair
label — its directional orientation on the hexagonal lattice. It used to be called
`sector_index`, but that name undersold it: it encodes the full phase-pair, not merely
a sector number. So as of Mark 2 the primary function is `phase_pair_sector`, with
`sector_index` retained as a backward-compatible alias so existing callers keep
working. The key property: rotating by 60° increments the phase-pair sector by exactly
one, and the label is preserved under all the symmetry operations the framework cares
about.

---

**Q: What is circle inversion here, exactly?**

A: Circle inversion about radius r maps a point to its "mirror" across the circle of
radius r — points inside the circle go outside and vice versa, while points exactly on
the circle stay put. Apply it twice and you're back where you started (it's an
*involution*). In this project the inversion is `ι_r` about **r²=12**, and it acts on
the integer shell labels in a beautifully exact way: shell N maps to shell **144/N**
(because 12² = 144). So shell 3 ↔ 48, 4 ↔ 36, 9 ↔ 16, and 12 maps to itself (the
self-dual boundary shell). All integer, all exact — `dual_shell_norm`,
`invert_sector_shell`, and `invert_label` implement it, and `verify_inversion_commutativity`
checks that it behaves.

---

**Q: Is circle inversion a rotation or a reflection? It sounds like one.**

A: Neither, and this distinction is load-bearing for the entire project. Inversion is
**conformal** (angle-preserving) but it is **not an isometry** — it does *not* preserve
distances. A rotation or reflection moves points around without changing the distances
between them; inversion stretches and shrinks. This is why the project draws a hard
**firewall**: you can use rotation to fold an *exact Euclidean distance* computation
(6×), but you can only use inversion to fold a *label/storage* computation — never a
metric one. More on that firewall in Part 6. If you remember one subtlety from this
whole document, make it this one.

---

## Part 3: Channels and Apples-to-Apples Methodology

---

**Q: What channel models are simulated?**

A: Three:
1. **Complex AWGN** — additive white Gaussian noise, the textbook baseline.
2. **2D impulsive** — occasional large-magnitude noise bursts on top of a Gaussian
   background; models bursty interference.
3. **Flat Rayleigh fading** — a randomly-scaled multiplicative channel with perfect
   channel state information (CSI) at the receiver; models a fading wireless link.

Each is implemented with exact, reproducible noise generation (fixed NumPy
`default_rng` seeds) so runs are repeatable.

---

**Q: What makes the hex-vs-square comparison fair?**

A: A stack of deliberate fairness preconditions, because an unfair comparison would
make the packing gain meaningless:
- **Matched order M** — same number of constellation points.
- **Matched average energy** — every constellation is normalized to unit average
  energy, so neither side wins by simply transmitting harder.
- **Common random numbers** — the *identical* noise/fading realizations are applied to
  both schemes, so the comparison isn't muddied by one scheme drawing luckier noise.
- **A true-Gray square-QAM baseline** — the square side uses a genuine Gray labeling,
  not a strawman.
- **Exact Clopper–Pearson confidence intervals** on the measured error rates.

The tests verify these preconditions *independently* of the comparison itself — e.g.
both constellations are checked to carry unit average energy, and the Eb/N0 → noise
variance mapping is checked against the closed-form relation.

---

**Q: Why is SER the headline and BER only secondary?**

A: **SER** (symbol error rate) asks "did the receiver pick the right constellation
point?" — it depends only on the *geometry*, not on how you assign bits to points.
**BER** (bit error rate) depends on the bit *labeling*. The hexagonal constellation's
labeling is **Gray-*like*** but not perfectly Gray (a perfect Gray code isn't always
available on a hex lattice), so it carries a small BER penalty. Reporting SER as the
headline keeps the comparison labeling-independent and honest; the BER penalty is
disclosed openly and never plotted as a win. SER leads; BER is the asterisk.

---

## Part 4: Exact Decoding (C1, C2)

---

**Q: What is the closed-form decoder, and how can it equal maximum-likelihood?**

A: For AWGN, the maximum-likelihood decision is simply "pick the nearest constellation
point." On the A₂ lattice there is a **closed-form nearest-point rule**: round the
received point into the lattice basis and check a small, provably-sufficient
**3×3 candidate window** around it. The closest of those candidates is *guaranteed* to
be the global nearest lattice point — so the result is **bitwise-identical** to an
exhaustive ML search over all M points. C1 is exactly this: the fast decoder equals
ML, verified over a dense grid and a Monte-Carlo stream with a target of **zero**
mismatches (and cross-checked against a wide brute-force search, including adversarial
"deep-hole" inputs that sit maximally far from any lattice point).

---

**Q: If it equals ML, what's the point? ML already exists.**

A: Cost. Exhaustive ML compares the received point against all M candidates — that's
O(M) per symbol. The closed-form decoder gets the *same answer* in **O(1)** on its
interior fast path, because the 3×3 window is a fixed-size check independent of M.
That's claim **C2**: constant-time decoding on the fast path. The win is in compute,
not in error rate (which is identical to ML, by construction).

---

**Q: So is it always O(1)? That sounds too clean.**

A: No — and the project is careful here. Points in the deep *interior* of the
constellation decode via the O(1) fast path. Points near the *boundary* can have a
nearest neighbor that falls outside the finite constellation, so those take an O(M)
exhaustive-ML **fallback** to stay exact. Therefore:
- **Worst case:** O(M) (a boundary symbol).
- **Amortized cost:** depends on how often the fast path applies — the **fast-path
  coverage** — which the simulation reports versus Eb/N0.

The clean story is "O(1) fast path with an exact O(M) safety net," not "O(1) always."

---

**Q: Does fast-path coverage actually improve with SNR?**

A: Yes, and that's the precondition C2 leans on. At higher SNR, received points land
closer to their true (mostly interior) symbols, so the fast path fires more often and
the amortized cost drops toward O(1). Simulation 01 reports coverage vs SNR
(`sim01_coverage_vs_snr.csv`), and a test pins the monotonic trend
(`test_fast_path_fraction_increases_with_snr`). The throughput speedup over ML grows
monotonically with M above a small crossover — bigger constellations, bigger win.

---

## Part 5: Packing Gain (C3)

---

**Q: What's the bottom line on hex vs square packing gain?**

A: It's real but modest, and it's channel-dependent. The careful summary:

| Channel | Verdict |
|---|---|
| **AWGN** | A resolved coding gain that grows from M=16 and **saturates near ~0.4 dB** at M≥64. |
| **Rayleigh** | **CI-limited** — point estimates positive, but the dB interval doesn't resolve at 100k trials. |
| **Impulsive** | Robustness **parity at the error floor** — the impulse floor masks any packing gain. |

The often-quoted **~0.6 dB** hexagonal packing figure is the *asymptotic* bound; it is
**not reached** at these finite M, so the docs report ~0.4 dB as what's actually
measured, not the textbook ceiling.

---

**Q: Why does Rayleigh only get "CI-limited" instead of a number?**

A: Because Rayleigh fading rolls off slowly (the error rate falls like ~1/SNR), so to
reach the low-SER targets you need very high Eb/N0 and enormous trial counts. At 100k
trials the confidence band on the dB gain is too wide to declare resolved — the point
estimate is positive, but honesty forbids reporting a crisp dB number the data can't
support. Instead, significance there is led by the **per-point McNemar test**, which
*does* resolve for M≥64 (just not M=16). The Rayleigh grid is extended out to ~44 dB
specifically so the slow roll-off can reach the targets at the larger M.

---

**Q: What is the McNemar + Holm test doing here?**

A: It's the per-point significance machinery. Because both schemes see the *same*
noise realizations (common random numbers), the right tool is a **paired** test:
**McNemar's test** compares, point by point, the cases where hex was right and square
wrong versus vice versa. Running it across many (channel, M, target) cells invites
false positives from multiple comparisons, so **Holm–Bonferroni** correction tightens
the α=0.05 threshold accordingly. The result is a defensible per-point "hex
significantly better here" flag (`sim02_significance_summary.csv`) rather than an
eyeballed curve.

---

**Q: Why is the impulsive channel only "parity"?**

A: Under heavy impulsive noise there's an **error floor** — a residual error rate
roughly `p·(1 − 1/M)` set by the impulse probability p, which no amount of clever
geometry removes. That floor masks the small packing gain, so the honest verdict is
robustness *parity at the floor*, not a hex win. Any genuine "hex < square" advantage
belongs to the pre-floor waterfall region; at the floor the two schemes tie. The
simulation even recovers the implied p from the measured floor to confirm the model.

---

## Part 6: Symmetry-Reduced Computation (C4)

---

**Q: What is the exact 6× reduction?**

A: A constellation metric — the pairwise squared-distance enumerator that yields
minimum distance, kissing number, and mean squared distance — has 6-fold rotational
symmetry. So you can compute it on **one sector** (one-sixth of the points), then
replicate, and reproduce the *identical* integer/rational quantity as the full
computation. Simulation 03 verifies this with `==` (not "approximately equal"), and
counts an exact **6× reduction in distance evaluations** (`sim03_symmetry.csv`). It's
a genuine compute saving that changes nothing about the answer.

---

**Q: And the firewall — what exactly can't I do?**

A: You **cannot** multiply the rotation 6× by the inversion fold to claim a bigger
combined Euclidean reduction. Here's why: rotation is an isometry, so it legitimately
folds a *distance* computation 6×. Inversion is conformal but **not** an isometry, so
it cannot fold a metric at all — it can only fold a discrete **label** enumerator. So
Simulation 03 reports two *separate* things in `sim03_inversion_reduction.csv`:
- The **Euclidean** enumerator folds by rotation **exactly 6×** (and the script
  explicitly demonstrates inversion is not an isometry — the firewall).
- A discrete inversion-invariant **label** enumerator folds by the full C₆×Z₂ group
  (rotation 6×, combined **10.5×**).

These two numbers are never multiplied into a single headline. The 10.5× is a
storage/label fold; the 6× is a metric fold; they live in different universes.

---

**Q: Where does 10.5× come from? That's a weird non-integer fold.**

A: It's not a clean 12× (the order of C₆×Z₂) because of the **self-dual boundary
shell**. Most shells pair up under inversion (N ↔ 144/N), so they fold cleanly. But
shell 12 maps to itself — it's its own dual — so it can't be folded by the inversion
half of the group the way the paired shells can. That boundary shell drags the
combined fold below 12×, landing at 10.5×. Honest accounting, not a typo.

---

## Part 7: Conflict-Free Parallel Recovery (C5)

---

**Q: What does the six-colouring have to do with parallelism?**

A: To denoise a signal spread over the lattice, you sweep over vertices updating each
from its neighbors. If two adjacent vertices update simultaneously they conflict. A
**proper colouring** assigns colours so that no two neighbors share one — so you can
update all vertices of a given colour *at once*, lock-free, with no conflicts. The
trihexagonal six-colouring gives a conflict-free schedule, and a colour-ordered
relaxation sweep denoises the lattice signal. On a GPU, all same-colour vertices
become one parallel batch.

---

**Q: How much does the GPU actually help?**

A: It depends on lattice size, and the project reports it as a measured systems result
rather than a fixed promise. On an RTX 4060, the speedup over a single-threaded CPU
baseline **widens with lattice size**: median **~13.8× at ~290k vertices**, crossing
unity at ~73k, and *below* unity for small lattices (where kernel-launch overhead
dominates). Simulation 04 prints a decisive `RAN_ON_CUDA = True/False` verdict, stamps
every CSV row with the device, and writes `sim04_provenance.json` (torch/CUDA build,
GPU name, memory) so the artifact self-certifies where it ran. Without a real GPU the
"GPU" column is just torch-on-CPU and `RAN_ON_CUDA = False` — you're told to report
*your* measured number, not the headline 13.8×.

---

**Q: I heard the six-colouring isn't actually rotation-equivariant. What happened?**

A: That's the Mark 2 honesty fix, and it's worth stating plainly. An earlier
description called the six-colouring "order-6-equivariant." That was an over-claim. The
truth:
- The underlying triangular-lattice **3-colouring** `(a−b) mod 3` (exposed as
  `three_coloring`) **is** order-6-equivariant — rotation permutes its three classes.
- The parity refinement `((a+b) mod 2)` that splits three colours into six **breaks**
  that equivariance. The six-colouring is **proper but not equivariant**.

Crucially, C5 only ever needed *properness* (for conflict-freedom), not equivariance —
so the claim itself stands untouched. Simulation 04 now records the corrected fact in
`sim04_coloring_equivariance.csv` (3-colouring: yes; six-colouring: no). Fixing the
wording cost nothing and bought correctness.

---

## Part 8: Rotation and Inversion Robustness (C6, C8)

---

**Q: What does "equivariant decode" (C6) mean?**

A: The decoder **commutes** with the order-6 rotation: if you rotate the input by 60°
and decode, you get the same answer as decoding first and rotating the label — the
sector index just permutes by +1. This is verified *exactly* on a real hex
constellation (`sim05_equivariance_check.csv`). It means the decoder treats all six
orientations consistently by construction, not by training or tuning.

---

**Q: What's the differential hexagonal scheme, and why is it the "hex DPSK"?**

A: Classic **DPSK** (differential phase-shift keying) encodes information in the
*difference* between successive symbols' phases, so a constant unknown phase offset
cancels out. The hexagonal analogue encodes information in successive **sector**
differences, making it invariant to any static phase ambiguity that's a multiple of
**π/3** (60°) — exactly the rotations the lattice can't otherwise distinguish. So if
your receiver has an unknown but fixed 60°·k phase rotation, the differential hex
scheme shrugs it off. Simulation 05 sweeps a phase offset crossing 60° to show it.

---

**Q: And C8 adds inversion on top of that?**

A: Yes. C8 is a **combined rotation+inversion differential codec**. It carries a pair
— a sector value in ℤ₆ *and* an inversion bit in ℤ₂ — as component-wise differences,
making it invariant under all **12** static C₆×Z₂ actions. So beyond a static 60°·k
phase ambiguity, it also absorbs a static **amplitude-inversion** ambiguity (the
inside-out flip). The critical detail: the inversion bit is a **discrete label state**,
never a Euclidean operation — consistent with the firewall from Part 6. Simulation 05
verifies recovery of *both* the sector data and the inversion bit under all 12 actions
with zero violations (`sim05_t24_check.csv`), using an independent RNG so it doesn't
perturb the pre-existing C6 outputs.

---

## Part 9: The Radial-Dual Constellation (C7)

---

**Q: What is the C7 object, concretely?**

A: A specific **filled hexagonal constellation** with M=42 points, built from the
shells {3, 4, 9, 12, 16, 36, 48}. It is the centerpiece of the "radial dual" idea
because it has three exact structural properties at once:
1. **Shell-complete** — each included shell is fully populated, giving exact C₆
   (order-6 rotational symmetry).
2. **Phase-pair-uniform** — every complete shell has the same number of points per
   sector; all six sectors are equally occupied.
3. **Inversion-paired** — closed under the exact circle inversion ι_r about r²=12,
   which maps each shell N to its integer-dual shell 144/N.

Put together, it's invariant under the full order-**12** C₆×Z₂ group.

---

**Q: Show me the shell pairing.**

A: Inversion about r²=12 sends shell N ↔ 144/N:

```
shell  3  ↔  shell 48     (3 × 48  = 144)
shell  4  ↔  shell 36     (4 × 36  = 144)
shell  9  ↔  shell 16     (9 × 16  = 144)
shell 12  ↔  shell 12     (12² = 144, self-dual boundary)
```

Three pairs plus one self-dual shell. The self-dual shell 12 is the one that sits *on*
the inversion circle and maps to itself pointwise — and it's exactly the reason the
inversion storage fold comes out to 10.5× rather than a clean 12× (Part 6).

---

**Q: Why insist it's order-12 (C₆×Z₂) and not the full order-24 group?**

A: Because over-claiming is the one sin this project won't commit. The full
centrosymmetric hexagonal point group D₆ₕ has 24 elements (rotations, reflections, and
inversion combined). The C7 object is verified invariant under the **rotation×inversion
subgroup** — C₆ (6 rotations) × Z₂ (identity + inversion) = **12** elements. It is
*not* claimed invariant under the full 24 (which would add the reflections). So the
docs say order-12 C₆×Z₂, full stop, rather than rounding up to a more impressive number.

---

**Q: What is the "folded decoder" and how big is the storage win?**

A: The phase-pair + inversion **folded** ML decoder stores only the **fundamental
domain** — the minimal slice of the constellation from which the symmetry regenerates
the rest — instead of a full per-point label table. It is still **bitwise-identical to
exhaustive ML** (every Euclidean decision uses true distances; folding only touches
*storage*). The storage ablation reports two *separate* factors:
- **6×** by rotation (phase-pair fold).
- **10.5×** by rotation+inversion (phase-pair + inversion fold).

Reported separately, never multiplied — same firewall as always. Simulation 06
verifies the bitwise-ML equivalence on a dense grid and a Monte-Carlo stream, and
notes the fast-path coverage is intentionally moderate here because the shell-complete
object has radial gaps.

---

## Part 10: Philosophy, Caveats, and Running It

---

**Q: Let's be blunt — does any of this beat ML on error rate?**

A: No, and it's not trying to. The closed-form and folded decoders *are* ML, so they
match ML's error rate exactly — never beating it. The wins are strictly:
- **Cost** — C1/C2 (O(1) fast-path decode), C4 (exact 6× metric fold), C5 (parallel
  GPU recovery), C7 (fundamental-domain storage folds).
- **Geometry** — C3 (packing gain), C6/C7/C8 (rotation/inversion robustness and the
  radial-dual structure).

If you ever see a claim that sounds like "TQF beats optimal decoding," it's a
misreading — the project is built specifically to *not* make that claim.

---

**Q: Why all the fuss about exact integer/rational arithmetic?**

A: Because the headline claims (C1, C4, C7) are **exactness** claims, and exactness
with floating point is a contradiction in terms. By keeping sectors, shells, colours,
inversion, and orbit reductions in integer/rational arithmetic, the tests can assert
`==` and `array_equal` with **zero tolerance**. A failure then means a real regression,
not a rounding drift. The Euclidean *decisions* still use true distances, but the
structural bookkeeping is exact by construction.

---

**Q: Is this production-ready?**

A: No. It's an experimental, after-hours research prototype that validates a set of
mathematical claims with reproducible scripts and tests. It is not a drop-in modem, it
hasn't been hardened for real RF front-ends, and the constellations and channels are
deliberately clean for the sake of falsifiable claims. Think proof-of-concept for
"hexagonal symmetry buys you exact, cheap, robust signal processing," not a shipping
product.

---

**Q: How do I reproduce everything?**

A: One command regenerates all seven studies and the figures:

```bash
./run_all.sh          # Linux/macOS
.\run_all.ps1         # Windows (also tees to results/run_all_console.log)
```

Or step through individually:

```bash
python src/simulation_01_hex_demod_correctness_and_latency.py   # C1, C2
python src/simulation_02_hex_vs_square_ber_packing_gain.py      # C3
python src/simulation_03_symmetry_reduced_metric_exact.py       # C4
python src/simulation_04_sixcoloring_denoise_gpu.py             # C5
python src/simulation_05_phase_rotation_robustness.py           # C6, C8
python src/simulation_06_phasepair_inversion_folded_decoder.py  # C1/C2 storage, C7
python src/simulation_07_radial_dual_constellation.py           # C7
python src/make_figures.py --format pdf
```

Each study writes its tables to `results/` (CSV + a `simNN_provenance.json`) and
prints a console summary. The full 129-test suite runs in a few seconds with
`python -m pytest tests/ -q` and needs no GPU.

---

**Q: What's the one number that depends on my hardware?**

A: Simulation 04's GPU speedup. It counts as a real result **only** when the printed
`RAN_ON_CUDA = True` verdict and `sim04_provenance.json` confirm an actual CUDA run. A
torch-on-CPU fallback is a reference backend, not the headline. So report the *measured*
GPU speedup from your own machine rather than the ~13.8× figure quoted here — that one
was an RTX 4060 at ~290k vertices, and your mileage will vary.

---

**Q: What's the single most important idea to walk away with?**

A: The **cost-vs-geometry firewall**, expressed through one geometric fact: rotation
preserves distance, inversion does not. Everything elegant in this project — the exact
6× metric fold that *can't* be combined with the 10.5× label fold, the folded decoder
that's still bitwise-ML, the differential codec where the inversion bit is a pure label
— flows from respecting that one boundary. The framework is generous with symmetry, but
it never lets a storage trick masquerade as a distance win.

---

**`QED`**

**Last Updated:** June 26, 2026<br>
**Version:** 1.1.0<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue at [tri-quarter-toolbox](https://github.com/nathanoschmidt/tri-quarter-toolbox) or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
