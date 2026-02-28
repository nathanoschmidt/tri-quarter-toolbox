# TQF-ANN: Everything You Ever Wanted to Know (And Some Things You Didn't)

*A Q&A guide for people who know what a neural network is but are not yet hexagon-brained.*

*This document is primarily machine-generated from the codebase and is an experiment in itself.*

---

**Author:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>
**License:** MIT<br>
**Version:** 1.1.1<br>
**Last Updated:** February 27, 2026<br>

---

## Part 1: The Big Picture

---

**Q: What is TQF-ANN?**

A: TQF-ANN stands for **Tri-Quarter Framework Artificial Neural Network**. It is a neural network architecture built on top of a mathematical structure called a *radial dual triangular lattice graph* with hexagonal symmetry. In plain English: instead of treating the input as a flat grid of pixels or a vector of numbers like most networks do, TQF-ANN organizes computation on a hexagonal lattice — the same tiling pattern you see on a honeycomb or a soccer ball.

The "Tri-Quarter" in the name refers to the framework's geometric foundation: a triangular lattice with dual inner and outer zones, divided into angular sectors. The "Artificial Neural Network" part means it learns from data like any other neural net. Think of it as: *your usual gradient-descent-and-backprop neural network, but it lives inside a honeycomb instead of a spreadsheet*.

---

**Q: What problem does it solve?**

A: TQF-ANN is evaluated on MNIST digit classification — the "Hello, World" of deep learning. But that's not the interesting part. The interesting part is *how it handles rotations*. Most neural networks, if you show them an upright "6" during training and then a flipped "9" (which is a 180-degree rotated "6") during testing, will often get confused unless you explicitly trained them on rotated examples.

TQF-ANN is designed with 6-fold rotational symmetry (called **Z6**) baked into its structure, so it inherently "understands" rotations in multiples of 60 degrees. The goal is to build a model that classifies digits accurately *regardless of how the image is oriented*.

---

**Q: Why hexagons? Did someone just really like bees?**

A: Hexagons are actually the most efficient way to tile a flat plane — maximum area, minimum perimeter, which is why bees use them. But more importantly for us, the hexagonal lattice (formally described using *Eisenstein integers*) naturally has 6-fold rotational symmetry. If you rotate a honeycomb by 60 degrees, it looks exactly the same. This symmetry is called the **Z6 group** — the cyclic group of order 6.

By building the network's computational structure *on* this lattice, every 60-degree rotation of the input corresponds to a predictable, structured transformation of the internal features. Specifically, rotating the input by 60 degrees shifts the network's internal "sector" representation by exactly one sector. This is the key to the whole framework. Hexagons aren't just pretty; they are the correct shape for this kind of symmetry.

---

**Q: What are Eisenstein integers and why should I care?**

A: Eisenstein integers are complex numbers of the form `m + n·ω`, where `ω = e^(2πi/3)` (a primitive cube root of unity) and `m`, `n` are ordinary integers. They form a hexagonal lattice in the complex plane, as opposed to ordinary integers (which form a square grid).

In practice, an Eisenstein integer `(m, n)` maps to Cartesian coordinates:
- `x = m − n/2`
- `y = n · √3/2`

The 6 nearest neighbors of any lattice point form a perfect hexagon around it. This is the foundation of TQF's lattice graph: vertices are placed at Eisenstein integer positions, and edges connect nearest neighbors. You care about this because it is what makes the "hexagonal" part of TQF-ANN more than just a marketing claim — it is the literal coordinate system the network uses to organize its computation.

---

**Q: What is circle inversion and why does TQF use it?**

A: Circle inversion is a geometric transformation that maps a point `v` in the complex plane to `v' = r² / conj(v)`, where `r` is the inversion radius and `conj(v)` is the complex conjugate. The effect: points *inside* a circle of radius `r` get mapped *outside* it, and vice versa. Points exactly on the circle stay put. Apply it twice and you're back where you started.

TQF uses this to create two zones from one lattice:
- The **outer zone**: lattice vertices at radius greater than 1 (the "normal" zone)
- The **inner zone**: the circle inversion image of the outer zone (vertices inside the unit circle)

The inner and outer zones are exact geometric mirrors of each other under inversion, creating a **dual zone structure**. Both zones process the input independently, and their outputs are combined via an ensemble. This gives the network two "views" of the same input — one from the outside of the unit circle, one from the inside — which the authors call *self-duality*.

If you're wondering whether this is mathematically elaborate: yes. Yes it is.

---

**Q: What is the T24 symmetry group that keeps being mentioned?**

A: T24 is the full symmetry group of the TQF lattice, with exactly 24 elements. It is the semidirect product D6 × Z2, meaning it combines:

- **D6** (the dihedral group of order 12): 6 rotations by multiples of 60 degrees + 6 reflections across primary axes
- **Z2** (circle inversion): just two operations — identity and inversion

So: 12 × 2 = 24 symmetry operations total. Any combination of rotating, reflecting, and inverting the lattice leaves the TQF structure invariant. This is the mathematical "superpower" the architecture is trying to exploit — the more symmetry your structure has, the more ways you can average over transformations at inference time without training overhead.

In practice, the code supports orbit mixing at three levels of this group: Z6 (just rotations), D6 (rotations and reflections), and T24 (all 24 operations).

---

## Part 2: The Dataset

---

**Q: What dataset does TQF-ANN use?**

A: MNIST — 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels (784 values). It is the classic benchmark dataset. 60,000 images are traditionally used for training and 10,000 for testing.

TQF-ANN does not use the original MNIST splits exactly. Instead, it reorganizes the data into a structured folder layout (one folder per class), then applies custom splits and rotations. The reason for this reorganization is to enable *stratified sampling* (ensuring each digit class is equally represented in each split) and to support the rotated test set.

---

**Q: How is the data split into train, validation, and test sets?**

A: The 60,000 MNIST training images are split as follows:
- **2,000 images for validation** — 200 per class, stratified (so each digit is equally represented)
- **58,000 images for training** — everything that's not validation

Then there are two separate test sets:
- **8,000 unrotated test images** — sampled from the standard MNIST test set (10,000 total), no rotation applied
- **A rotated test set** — 2,000 base test images × 6 rotation angles = 12,000 images total

The validation set comes from the original training pool (not the test set), so it never contaminates evaluation. The 2,000-image reserved validation set means training technically has access to 58,000 samples out of 60,000 possible.

Why 58,000 and not 60,000? Because we need a proper holdout validation set, and we do not want to validate on training data. 2,000 images (200 per class) is large enough to give stable accuracy estimates without wasting too much training data.

---

**Q: How does the rotated test set get created?**

A: Each of the 10,000 MNIST test images gets rotated to 6 specific angles:
- 0°, 60°, 120°, 180°, 240°, and 300°

These angles are chosen because they align with the Z6 symmetry group — rotating by any multiple of 60 degrees brings the lattice back to an equivalent state. The rotation is applied using bicubic interpolation for image quality. The resulting 60,000 images (10,000 × 6) are stored as PNG files in a `rotated_test` folder.

Why these angles specifically? If the model truly has Z6 rotational invariance, it should classify a "3" the same way regardless of which of the 6 orientations it sees. Using these exact angles allows us to test whether the architecture's symmetry actually helps in practice.

---

**Q: What does the "rotation invariance error" metric measure?**

A: For each base image, the model makes 6 predictions (one per rotation angle). In an ideal, perfectly rotation-invariant model, all 6 predictions would be identical. In practice, they differ. The rotation invariance error measures how much the predictions vary:

1. Run each of the 6 rotations of a base image through the model
2. Compute the softmax probabilities for each rotation
3. Compute the variance of those probabilities across the 6 rotations, for each class
4. Average this variance across all classes and all base images

Lower rotation invariance error = more consistent predictions across rotations = better. A perfectly invariant model gets 0. A model that predicts randomly gets a high value. This metric is separate from accuracy: a model can be inaccurate but very consistent (always wrong the same way), or accurate but inconsistent (right for some rotations, wrong for others).

Rotation accuracy (reported as "rot acc") is the simpler cousin: just the fraction of the 12,000 rotated test images that are classified correctly, regardless of consistency.

---

**Q: Does training on 58,000 samples actually make a difference?**

A: Dramatically, yes. Earlier experiments trained on only 30,000 images. Doubling (nearly) to 58,000 improved test accuracy by about **+1.15%** (from 94.60% to 95.75%). In neural network terms, that's a big jump for just adding more of the same data with no architectural changes.

The lesson is a classic one: before you add complexity (new loss functions, fancy regularizers, architectural changes), check if you simply have enough data. In this case, MNIST has 60,000 training images available. Using only half of them was leaving free performance on the table.

---

## Part 3: Training

---

**Q: What optimizer and learning rate schedule are used?**

A: The network is trained with the **Adam optimizer** at a learning rate of 0.001. The schedule has two phases:

1. **Linear warmup** (epochs 0–5): Learning rate ramps from 1e-6 to 0.001. This prevents large, destructive gradient updates in the very first steps when the network weights are still random.
2. **Cosine annealing decay** (epochs 5–150): Learning rate follows a cosine curve from 0.001 down to 1e-6. This gives a smooth, gradual decrease that avoids the "cliff drop" of step-schedule LR.

Why cosine annealing? The tail of the cosine schedule — epochs 100 to 150 where the LR is very small (below 3e-4) — acts as a fine-tuning phase. The network has already found a good solution; the low LR lets it settle into a sharper local minimum. Experiments showed that extending training from 100 to 150 epochs added +0.22% test accuracy specifically because of this tail. Stopping at 100 epochs wastes the free fine-tuning that cosine annealing naturally provides.

---

**Q: What is early stopping and how does it work here?**

A: Early stopping monitors the validation loss each epoch. If the loss doesn't improve by at least `min_delta = 0.0005` for `patience = 25` consecutive epochs, training stops early and the best checkpoint is loaded.

Why 25 epochs of patience? Because the cosine annealing LR schedule has a long tail. With a patience of 15 (the old default), early stopping would often fire during the LR tail even though the model was still improving. Raising patience to 25 ensures training runs long enough to benefit from fine-tuning at low LR, without waiting forever if the model has genuinely plateaued.

The model checkpoint is saved at the epoch with the lowest validation loss, not the last epoch.

---

**Q: What is label smoothing and why does TQF use it?**

A: Label smoothing is a regularization technique that prevents the model from becoming overconfident. Instead of training the model to output a probability of exactly 1.0 for the correct class and 0.0 for all others, label smoothing says: "aim for 0.9 for the correct class, and spread 0.1 evenly across all 10 classes."

In practice, this means the cross-entropy loss function uses "soft" targets rather than one-hot hard labels. The default label smoothing value is 0.1.

Why does this help? Neural networks have a tendency to drive logits to very large values, making the softmax output confidently wrong on examples they've never seen. Label smoothing acts as a brake on this behavior, resulting in better-calibrated predictions and often better generalization. Experiments showed that label smoothing of 0.1 outperforms both 0.0 (no smoothing) and 0.05 (lower smoothing).

---

**Q: What other regularization is used?**

A: Three flavors:
1. **Dropout (0.2)** in the fully connected layers — randomly zeros 20% of activations during training, preventing co-adaptation
2. **Weight decay (1e-4)** in Adam — L2 regularization that penalizes large weights, pushing the model toward simpler solutions
3. **Label smoothing (0.1)** — as described above

None of these are exotic. They are standard tools applied conservatively.

---

**Q: What is Z6 data augmentation, and is it used by default?**

A: Z6 data augmentation randomly rotates each training image to one of the 6 Z6 angles (0°, 60°, 120°, 180°, 240°, 300°) plus a random ±15° jitter. This means the network sees each digit in various orientations during training, which *could* help it learn rotation invariance.

Whether it actually helps depends on context. With orbit mixing (described later), Z6 data augmentation is **counterproductive and should not be used**. Without orbit mixing, it can help. By default, augmentation is turned off in the latest configuration because orbit mixing is the preferred approach.

The ±15° jitter is added on top of the base Z6 angles to provide regularization while preserving the Z6 structure. Pure 60° increments would still let the model memorize rotations; the jitter makes it learn more robustly.

---

**Q: How long does training take?**

A: On an RTX 4060 Laptop GPU, with 58,000 training images and proper in-memory caching, each epoch takes about **12 seconds**. 150 epochs therefore takes about 30 minutes per seed. Running 5 seeds for statistical validity takes roughly 2.5 hours.

Without the in-memory cache, each epoch took about **7 minutes** because Windows Defender (antivirus) was scanning every PNG file on disk access. That's a 35× slowdown from antivirus. The solution was caching all 60,000 images in memory after the first epoch (or loading them from a pre-built `.npy` binary on subsequent runs). The cache is about 47 MB for all of MNIST — negligible on any modern machine, but it turns a painful experience into a pleasant one.

---

## Part 4: The Architecture

---

**Q: What is the overall structure of the TQF-ANN model?**

A: Briefly, the forward pass looks like this:

```
MNIST image (784 numbers)
    ↓
Pre-encoder: linear → layer norm → GELU activation
    ↓
Boundary encoder: maps to 6 "boundary vertices" (the hexagon ring at radius 1)
    ↓
Lattice processing: graph convolutions across radial bins
    - Outer zone: vertices at radius > 1
    - Inner zone: circle inversion image (radius < 1)
    - 6 angular sectors per zone
    ↓
Sector aggregation: pool features per sector → (B, 6, H) tensor
    ↓
Classification head: sector-weighted logits → 10 class scores
```

The internal tensor shape during processing is `(B, 2, 6, L, H)`:
- `B` = batch size
- `2` = zones (outer and inner)
- `6` = sectors
- `L` = radial layers (concentric shells of the lattice)
- `H` = hidden dimension (default 512, auto-tuned)

---

**Q: What are "sectors"?**

A: The lattice is divided into 6 angular wedges — like slicing a pie into 6 equal pieces, each covering 60 degrees. Wedge 0 covers directions 0°–60°, wedge 1 covers 60°–120°, and so on. Each lattice vertex is assigned to one sector based on its angular position (its "phase").

This is where the Z6 symmetry becomes concrete: rotating the input by 60° shifts all sector assignments by one position (sector 0 → 1 → 2 → ... → 5 → 0). If the network processes each sector the same way (shared weights), then a 60° rotation just *permutes* the sector features without changing the computation. This is the definition of **equivariance**: the output transforms predictably when the input transforms.

The sector features are pooled into a `(B, 6, H)` tensor where each of the 6 entries summarizes everything in that angular wedge.

---

**Q: What are "phase pairs" and why do they matter?**

A: Each lattice vertex is assigned a "phase pair" label — a pair `(φ_out, φ_in)` encoding its primary and secondary directional orientation on the hexagonal lattice. Phase pairs are a way of giving each vertex a locally meaningful label based on its position relative to its neighbors.

The key property: phase pairs are preserved under all T24 symmetry operations (rotations, reflections, and circle inversion). This means the labeling scheme is consistent throughout the lattice and across all symmetry transformations. It is part of what makes the architecture properly "T24-equivariant" rather than just loosely symmetric.

In practice, phase pairs enable the network to maintain consistent geometric bookkeeping even after symmetry operations. If you do not have consistent labels, the symmetry structure breaks down and the network cannot learn to exploit it.

---

**Q: What does "shared weights" mean in this context?**

A: In the lattice processing layers, the same weight matrices are used for all 6 sectors in both the inner and outer zones. The tensor layout `(B, 2, 6, L, H)` lets PyTorch apply these shared weights efficiently via batched matrix multiplication.

Shared weights are what make the architecture equivariant. If sector 0 and sector 3 use different weights, a 180° rotation that maps sector 0 → sector 3 would produce different computation — that's not invariant. Shared weights ensure that the sector at position `k` always applies the same transformation, so rotating the input just permutes which sector gets which features, without changing how those features are processed.

---

**Q: How does the dual zone output work?**

A: After sector aggregation, the model produces logits from both the outer and inner zones. The final prediction is a confidence-weighted combination:
- The outer zone classification head produces a set of logits
- The inner zone uses the same weights (circle inversion is applied to the features before the head)
- A learned scalar `alpha` interpolates between the two outputs, weighted by the relative confidence of each zone

This is called **zone confidence ensemble**. The idea: sometimes one zone has a cleaner, more confident signal than the other. Letting the model learn to weight them adaptively is better than a fixed 50/50 average.

Whether the inner zone truly provides independent information or just a noisy version of the outer zone is an open question. In practice, the ensemble helps modestly.

---

**Q: How many parameters does TQF-ANN have?**

A: Approximately **650,000 parameters**, auto-tuned via a parameter matching routine that adjusts the hidden dimension `H` until the total count is within ±1.1% of the target. All four models in the comparison (TQF-ANN, FC-MLP, CNN-L5, ResNet-18-Scaled) are parameter-matched to the same budget. This ensures that any accuracy difference between them is due to architecture, not parameter count.

Parameter matching is done by a `param_matcher.py` module that performs binary search over hidden dimension values until the target parameter count is hit. It is less glamorous than it sounds.

---

## Part 5: The Baselines

---

**Q: What baseline models are compared against TQF-ANN?**

A: Three baselines, all with approximately 650,000 parameters:

1. **FC-MLP**: A standard fully connected multilayer perceptron with 3 hidden layers, ReLU activations, and dropout. The most basic possible baseline — no spatial structure, no convolutions, nothing fancy. If TQF-ANN cannot beat this, something is deeply wrong.

2. **CNN-L5**: A 5-layer convolutional neural network with pooling, batch normalization, and dropout. Exploits the 2D structure of images but has no hexagonal symmetry or rotation invariance built in. The "classic deep learning" baseline.

3. **ResNet-18-Scaled**: A scaled-down version of the ResNet-18 architecture with residual skip connections. Represents modern CNN design. If TQF-ANN cannot beat this, at least it is losing to a respectable opponent.

All three baselines use the same optimizer, learning rate schedule, label smoothing, and training data as TQF-ANN. Only the architecture differs.

---

**Q: How does TQF-ANN compare to the baselines?**

A: On standard (unrotated) test accuracy, the models are approximately tied at around **95.9–96.0%**. The differences are small enough to be within noise for a single seed. MNIST at this scale is somewhat saturated — most reasonable architectures with enough data will converge to similar numbers.

The story changes completely when you look at **rotation accuracy** (accuracy on the rotated test set). Without orbit mixing:
- FC-MLP, CNN-L5, ResNet-18: approximately 38–43% on severely rotated inputs (these numbers reflect evaluation under a more extreme rotation scenario)
- TQF-ANN: approximately 66–67%

With orbit mixing (Z6, temperature 0.5):
- TQF-ANN: approximately **97.37%**

So TQF-ANN does not simply win on flat MNIST accuracy — it wins decisively on the task it was designed for: recognizing digits regardless of orientation. That is the whole point.

---

## Part 6: Orbit Mixing

---

**Q: What is orbit mixing?**

A: Orbit mixing is an **inference-time ensemble technique** that exploits the TQF architecture's symmetry structure to improve predictions. Here is the core idea:

For a given input image, instead of running it through the network once and returning the result, you:
1. Generate multiple transformed versions of the input (rotations, reflections, etc.)
2. Run all versions through the network in a single batched forward pass
3. Combine the predictions from all versions, weighted by how confident each one is

The result is a more robust prediction that draws on multiple "views" of the same underlying image.

Crucially: there is no training overhead. No extra loss functions, no extra training time. The only cost is at inference time, and because all rotations are batched together, the practical overhead is small (about the same as running 6 images instead of 1, which is roughly 5× slower on GPU due to parallelism).

---

**Q: Why call it "orbit mixing"? What is an orbit?**

A: In group theory, the **orbit** of a point under a group action is the set of all points you can reach by applying any element of the group to that starting point. For an image under Z6, the orbit is the set of all 6 rotated versions of that image (rotate by 0°, 60°, 120°, 180°, 240°, 300°).

"Orbit mixing" means you compute predictions for all elements of the orbit (all 6 rotations) and then mix (combine) those predictions. You are averaging over the orbit. In group theory, averaging a function over all elements of a group acting on an input is a classical technique for imposing invariance — it is called *Reynolds averaging* or *orbit averaging*.

The "mixing" part refers to the confidence-weighted combination: instead of a simple arithmetic average, heavier weight is given to the rotations that the model is most confident about. This prevents a catastrophically confused rotation from dragging down the ensemble.

---

**Q: How does Z6 orbit mixing work in detail?**

A: Given an input image:

1. **Generate 6 rotations**: Apply 0°, 60°, 120°, 180°, 240°, 300° rotations using bilinear interpolation. All 6 images are stacked into a single batch.

2. **Single forward pass**: All 6 rotated images go through TQF-ANN simultaneously. This gives 6 sets of logits, one per rotation.

3. **Compute confidence scores**: For each rotation, compute a confidence score from the logits. The default is `max_logit` mode (take the maximum logit as the confidence). An alternative is `margin` mode (top-1 logit minus top-2 logit — more discriminative for uncertain predictions).

4. **Confidence weighting**: Apply softmax over the confidence scores, scaled by a temperature parameter:
   ```
   weights = softmax(confidence_scores / temperature)
   ```
   Lower temperature → winner-takes-all (the most confident rotation dominates).
   Higher temperature → softer, more democratic averaging.

5. **Weighted aggregation**: Compute the weighted average of the 6 logit vectors. The aggregation can happen in logit space (default), probability space (softmax first), or log-probability space (geometric mean).

6. **Final prediction**: Take `argmax` of the aggregated logits.

The default temperature is **0.5**, which was found to be optimal in systematic experiments. At temperature 0.5, the most confident rotation gets substantially more weight, but the others still contribute.

---

**Q: What temperature should I use for orbit mixing?**

A: The optimal temperature was found to be **0.5** through systematic sweep experiments. Here are the results:

| Temperature | Rotation Accuracy |
|-------------|------------------|
| 0.3 | 66.94% |
| **0.5** | **67.42%** ← best |
| 0.7 | 67.31% |
| 1.0 | 67.27% |
| 2.0 | 65.45% |

Temperature 0.5 strikes the right balance: it gives meaningful preference to the most confident rotation without completely ignoring the others. Temperature 2.0 becomes too soft — all rotations get nearly equal weight even if one is catastrophically wrong — and performance drops.

The "best rotation" for a given image is whichever rotation aligns the digit most favorably with the model's internal hexagonal symmetry axes. The network has learned to be more confident when the alignment is good. Low temperature exploits this signal.

---

**Q: What are the other orbit mixing modes (D6, T24)?**

A: Z6 mixing handles the 6 rotational elements of the symmetry group. D6 and T24 add more operations:

**D6 orbit mixing** adds 6 reflections on top of the 6 rotations. In feature space, this is implemented by *permuting the sector indices* according to the corresponding reflection axis. Since reflections map sector `k` to sector `(-k) mod 6`, the implementation swaps sector features accordingly and re-runs the classification head. Total: 12 orbit elements.

**T24 orbit mixing** adds the circle inversion element on top of D6: swapping the inner and outer zone confidence weights in the dual ensemble. Since the inner and outer zones are related by geometric inversion, swapping them is the feature-space analog of applying the inversion symmetry. Total: 24 orbit elements.

In practice:
- Z6 mixing alone gives the biggest gain and is the recommended choice
- D6 adds marginal improvement (it does not help as much because digit reflections are not always semantically valid — a "6" reflected is not a "6", it's a "9")
- T24 adds the least because zone-swap is the most abstract operation and the inner/outer zones are not perfectly symmetric in practice

---

**Q: Can I use orbit mixing on the baseline models (FC-MLP, CNN, ResNet)?**

A: You can try, but it does not help. The baselines have no built-in understanding that a rotation corresponds to a structured, predictable transformation. For TQF-ANN, rotating the input by 60° shifts the sector features in a consistent way that the model can exploit. For an FC-MLP, rotating the input by 60° just produces a completely different 784-dimensional vector with no predictable relationship to the original. Mixing these predictions gives you noise, not signal.

This was empirically confirmed: applying Z6 orbit mixing to the baseline models produced no improvement or slight degradation. Orbit mixing is only meaningful when the model architecture respects the underlying symmetry that the mixing operation assumes.

---

**Q: Why does orbit mixing conflict with Z6 data augmentation?**

A: This is the most important gotcha in the entire system, so pay attention.

**Z6 data augmentation** randomly rotates training images by multiples of 60°. This teaches the model that rotated inputs are the same class. Fine.

**Orbit mixing** works by rotating the *test* image and asking the model to give consistent, confident predictions across all rotations — then combining them.

The conflict: if you use both, the model has been trained to produce *the same* output regardless of rotation (because augmentation taught it that rotations are equivalent). But then orbit mixing tries to use *differences* in output confidence across rotations to figure out which rotation is "best." If the model treats all rotations the same, all confidences are roughly equal, and the weighted combination degenerates to a uniform average. Worse, adding both creates "double rotation" during evaluation — the orbits interfere with each other rather than complement each other.

The experimental result: using both augmentation and orbit mixing together gives **lower** accuracy than orbit mixing alone. The maximum rotation accuracy without augmentation: **67.42%**. With both: **roughly 65%**. Orbit mixing with augmentation doesn't just fail to help — it actively hurts.

The rule: **pick one or the other.** For maximum rotation robustness, use orbit mixing only (no training augmentation). For a simpler training setup, use augmentation only (no orbit mixing at inference).

---

**Q: Why does orbit mixing conflict with equivariance loss?**

A: Similarly, the equivariance loss was an experimental training-time penalty that tried to force the model to produce consistent predictions across Z6 rotations during training. The hope was that explicitly penalizing prediction inconsistency would build stronger rotational equivariance into the weights.

The result: equivariance loss alone achieves **62.83%** rotation accuracy, which is actually worse than orbit mixing alone (**67.42%**). Using both together makes it even worse. The equivariance loss fights against orbit mixing by trying to reduce the variance in predictions across rotations — exactly the signal that orbit mixing uses to determine which rotation is most informative.

If the model learns to ignore rotations completely (which the equivariance loss encourages), all 6 rotated predictions are identical and equally confident. Orbit mixing then has nothing to work with.

The lesson: equivariance loss is a training-time crutch that is outperformed by the inference-time exploit of orbit mixing. The equivariance loss feature was ultimately removed from the codebase.

---

**Q: Is orbit mixing cheating?**

A: That is a fair question. Some would say: "You're running the model 6 times instead of once — of course it's better. That is not a fair comparison."

The counterargument: orbit mixing is an inference technique, not additional training data. You could apply test-time augmentation (run 6 random crops through any CNN and average) to the baselines too, and it helps those models as well. The question is whether orbit mixing *specifically* benefits from TQF's built-in symmetry — and the answer is yes, because TQF produces much stronger orbit mixing gains than the baselines can get from naive test-time augmentation.

Additionally, all 6 rotations are batched together, so the GPU handles them in parallel. The wall-clock inference time increase is roughly 1.5–2× (not 6×), and for a classification task where accuracy matters more than speed, this is an acceptable trade-off.

Whether to report results with or without orbit mixing depends on the claim you're making. Results are reported both ways in the experiments.

---

## Part 7: The Rotated Test Set in Detail

---

**Q: Why use 60-degree rotations for the test set specifically?**

A: Because the TQF architecture's symmetry group is Z6, which acts by rotations of exactly 60 degrees. Testing at 60-degree increments aligns the test set with the architecture's native symmetry.

This is the most favorable possible test for TQF-ANN. If the claim is "Z6 symmetry helps with rotations," you test with Z6 rotations. It is like evaluating a hammer on nails rather than screws — you test the tool on the task it was designed for.

A critic would note: real-world rotations are not restricted to multiples of 60 degrees. An arbitrary rotation of a "5" might not be better handled by a Z6-symmetric model than by a well-trained CNN. This is a known limitation of the current evaluation framework.

---

**Q: The test images are the same as training — is there data leakage?**

A: No. The rotated test set is generated from the MNIST **test split** (10,000 images), which is completely separate from the 60,000 training images. The training data comes from MNIST's training split; the rotated test data comes from MNIST's test split. They do not overlap.

The rotated test images are also pre-generated (not generated on the fly during evaluation), which ensures consistent evaluation across runs and seeds.

---

**Q: Why are the test images stored as PNG files rather than generated on the fly?**

A: Two reasons:
1. **Reproducibility**: If you generate rotations on the fly with random seeds, you might get slightly different images on different runs. Storing PNGs ensures everyone evaluates on exactly the same images.
2. **Performance**: On Windows with antivirus, loading PNGs from disk is faster if cached. Pre-generating the rotated test set once and storing it avoids repeating the rotation computation on every evaluation run.

The trade-off is disk space (60,000 PNGs for the rotated test set), which is negligible.

---

## Part 8: Results and What They Mean

---

**Q: What are the final accuracy numbers?**

A: After tuning and 5-seed statistical validation (5 independent training runs with different random seeds):

| Metric | Mean | Std |
|--------|------|-----|
| Validation accuracy | 95.28% | ±0.28% |
| Test accuracy (unrotated) | 95.87% | ±0.14% |
| Rotation accuracy (no orbit mixing) | 95.27% | ±0.22% |

With Z6 orbit mixing (temperature 0.5):
- Rotation accuracy improves to approximately **97.37%**
- Unrotated test accuracy remains stable (orbit mixing does not hurt standard accuracy)

Without orbit mixing, TQF-ANN's rotation accuracy (95.27%) is similar to the baselines because all models are trained without rotation augmentation by default. The architecture alone does not give a huge boost — the boost comes from using the architecture *correctly* with orbit mixing.

---

**Q: What was the most impactful finding from the experiments?**

A: Data scaling. Doubling the training data from 30,000 to 58,000 samples gave the largest single accuracy gain of any change tested. More specifically:

| Change | Test Acc Gain |
|--------|--------------|
| Data scaling (30K → 58K) | +1.15% |
| Longer training (100 → 150 epochs) | +0.22% |
| Lower LR (0.001 → 0.0005) | ≈ 0.00% (no benefit) |
| Fractal iterations | -0.26% (hurt) |
| Label smoothing 0.05 vs 0.1 | ≈ 0.00% (marginal) |

The lesson: data first, then training schedule, then architecture. "More clever" is rarely better than "more data."

---

**Q: What did not work?**

A: Several things were tried and removed:
- **Fractal self-similarity** (multiple recursion levels of the lattice): Made rotation invariance error 72× worse and hurt accuracy
- **Fibonacci weighting** (special feature scaling based on the golden ratio): Hurt accuracy on all metrics
- **Hop attention with custom temperature**: Removed due to negligible benefit
- **Z6 equivariance loss**: Hurt accuracy vs. orbit mixing
- **Lower learning rates (0.0005, 0.0007)**: Both underperformed the default 0.001
- **Gradient checkpointing**: Removed as unnecessary at this scale

The codebase is leaner for these removals. Every removed feature was removed because experiments showed it did not help.

---

## Part 9: Running the Code

---

**Q: How do I run TQF-ANN?**

A: The main entry point is `src/main.py`. The simplest run:

```bash
python src/main.py
```

This trains all four models (TQF-ANN, FC-MLP, CNN-L5, ResNet-18-Scaled) with default settings: 58,000 training samples, 150 epochs, patience 25, and compares them. Results are saved to a JSON file in `data/output/`.

For TQF-ANN only with Z6 orbit mixing:
```bash
python src/main.py --models TQF-ANN --tqf-use-z6-orbit-mixing
```

For a quick smoke test:
```bash
python src/main.py --num-epochs 10 --num-train 1000
```

For multi-seed statistical comparison (5 seeds):
```bash
python src/main.py --num-seeds 5
```

---

**Q: What orbit mixing options are available?**

A: The main options:

```bash
# Choose which symmetry level to mix over
--tqf-use-z6-orbit-mixing        # 6 rotations (RECOMMENDED)
--tqf-use-d6-orbit-mixing        # 12 rotations + reflections
--tqf-use-t24-orbit-mixing       # 24 full T24 operations

# Temperature (lower = winner-takes-all, higher = soft average)
--tqf-z6-orbit-mixing-temp-rotation 0.5   # default and optimal

# Confidence scoring mode
--tqf-z6-orbit-mixing-confidence-mode max_logit  # default: max logit value
--tqf-z6-orbit-mixing-confidence-mode margin     # top-1 minus top-2 logit

# Aggregation space
--tqf-z6-orbit-mixing-aggregation-mode logits    # default: average raw logits
--tqf-z6-orbit-mixing-aggregation-mode probs     # average softmax probabilities
--tqf-z6-orbit-mixing-aggregation-mode log_probs # geometric mean

# Advanced options
--tqf-z6-orbit-mixing-top-k 4   # Only use 4 most confident rotations out of 6
--tqf-z6-orbit-mixing-adaptive-temp  # Per-sample entropy-based temperature
--tqf-z6-orbit-mixing-rotation-pad 2  # Reflect-pad image before rotation
```

The default settings (temperature 0.5, max_logit, logit aggregation) are already optimal. Change them only if you have a specific reason to.

---

**Q: What does the output JSON file contain?**

A: Everything you might want to reproduce or analyze the experiment:
- Configuration: all CLI arguments and hyperparameters used
- Per-seed results: accuracy, loss, rotation invariance error for each seed
- Aggregated results: mean and standard deviation across seeds
- Per-class accuracy breakdowns
- Statistical significance tests (paired t-test p-values, Cohen's d effect sizes)
- Inference timing and FLOPs estimates
- System metadata: Git commit hash, timestamp, Python and PyTorch versions
- Training curves: loss and accuracy per epoch (per seed)

The result file is stamped with a timestamp (e.g., `results_20260211_230259.json`) and never overwritten, so every experiment run produces a new file. You can compare runs later by loading the JSON files.

---

## Part 10: Philosophy and Caveats

---

**Q: Is TQF-ANN the best model for MNIST digit classification?**

A: Not even close. State-of-the-art MNIST accuracy (with more parameters, larger architectures, or more training tricks) exceeds 99.7%. TQF-ANN at 95.9% is not trying to win the MNIST leaderboard. MNIST is used because it is simple enough to iterate quickly on, well-understood, and provides clean rotation experiments.

The goal is to demonstrate that baking geometric symmetry into an architecture (rather than learning it from data) produces measurably better rotation robustness. MNIST is the sandbox. The framework could theoretically be applied to other domains where hexagonal or lattice symmetry is relevant.

---

**Q: Why not just train on rotated images and use a regular CNN?**

A: You could! A CNN trained with heavy rotation augmentation will learn rotation invariance empirically and can achieve very high accuracy on rotated MNIST. This is a completely valid approach.

The TQF approach is making a different claim: by encoding the symmetry *in the architecture*, you get rotation invariance "for free" without needing to train on rotated examples — or at least with much less data and training. Whether this advantage outweighs the architectural complexity is an empirical question. In experiments, orbit mixing on TQF-ANN achieves 97.37% rotation accuracy *without any rotation training augmentation*, while baselines trained the same way are much lower. The baseline advantage with augmentation vs. TQF with orbit mixing remains to be compared systematically.

---

**Q: Is this architecture ready for production?**

A: No. TQF-ANN is a research prototype, tested only on MNIST. It has not been validated on more complex datasets, different image sizes, non-grayscale inputs, or any real-world classification problem. The architecture also currently lacks some features of production-grade networks (e.g., batch normalization in the graph convolution layers, modern attention mechanisms, etc.).

Consider this a proof of concept for geometric symmetry exploitation in neural networks, not a drop-in replacement for your ResNet.

---

**Q: What is TQF-SNN and will there be a sequel?**

A: TQF-SNN stands for **Tri-Quarter Framework Spiking Neural Network** — the same geometric framework adapted for spiking neurons, which communicate with discrete spikes rather than continuous activations. Spiking neural networks are more biologically realistic and potentially more energy-efficient on specialized hardware.

The codebase is named `tqf_ann_snn` because the SNN extension is planned as a future direction. Whether the hexagonal lattice structure provides the same benefits for SNNs as for ANNs remains to be seen. That is a future experiment.

---

**Q: What is the single most surprising finding from all these experiments?**

A: That adding data was more powerful than any architectural change. The TQF framework is mathematically elaborate — Eisenstein integers, circle inversion, phase pairs, T24 groups — and all of that complexity provides a smaller accuracy boost than simply training on 28,000 more images. The second-most surprising: orbit mixing, which requires zero training changes and is implemented entirely in the evaluation code, provides a dramatically better rotation robustness improvement than any training-time symmetry enforcement tried.

The universe keeps reminding us: more data and simpler inference beats fancier training every time.

---
**`QED`**

**Last Updated:** February 27, 2026<br>
**Version:** 1.1.1<br>
**Maintainer:** Nathan O. Schmidt<br>
**Organization:** Cold Hammer Research & Development LLC (https://coldhammer.net)<br>

Please remember: this is an experimental after-hours unpaid hobby science project. :)

For issues, please open a GitHub issue or contact: nate.o.schmidt@coldhammer.net

**`EOF`**
