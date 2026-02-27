"""
config.py - Centralized Configuration for TQF-NN Experiments

This module defines all hyperparameters, dataset sizes, and architectural
constants for the Tri-Quarter Framework Neural Networks (TQF-NN) project
and baseline model experiments.

Key Features:
- Reproducibility: Fixed random seeds and multi-seed experiment configuration
- Dataset Configuration: Stratified sampling sizes for train/val/test splits
- Training Hyperparameters: Learning rate, optimizer settings, regularization
- TQF Architecture: Lattice geometry, symmetry groups (Z6/D6/T24), fractal parameters
- Dual Metrics: Hyperbolic distance computations and duality verification
- Parameter Matching: Fair "apples-to-apples" comparison constraints (~650K params)
- Hardware Settings: CUDA optimization, data loading parallelism

Each constant is documented with:
  - WHAT: Clear definition of the parameter
  - HOW: Usage instructions and valid value ranges
  - WHY: Scientific rationale and importance for the experiment

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026
"""

# MIT License
#
# Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from typing import List

###################################################################################
# OUTPUT CONFIGURATION ###########################################################
###################################################################################

# DEFAULT_RESULTS_DIR
# -------------------
# WHAT: Default directory for persistent experiment result files (JSON + TXT).
#
# HOW: Results are saved incrementally after each seed completes, so partial
#      results survive crashes or interruptions. The directory is relative to the
#      project root (parent of src/).
#      - Default: "data/output" (resolved to absolute path at runtime)
#      - CLI override: --results-dir PATH
#      - Disable saving: --no-save-results
#
# WHY: Persistent result logging ensures long-running experiments are never lost.
#      The data/output/ directory is gitignored (except for .gitkeep) so results
#      don't pollute version control.
DEFAULT_RESULTS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'output'
)

###################################################################################
# REPRODUCIBILITY CONSTANTS #######################################################
###################################################################################

# NUM_SEEDS_DEFAULT
# -----------------
# WHAT: Number of independent random seeds used for multi-seed experiments.
#
# HOW: Controls statistical robustness through repeated training runs. Each seed
#      produces a unique weight initialization and data shuffling sequence. Results
#      are aggregated (mean +/- std) across all seeds for significance testing.
#      - Typical range: 1 (fast single run) to 10 (robust statistics)
#      - CLI override: --num-seeds N
#      - Single runs: Set to 1 for rapid prototyping
#      - Full benchmarks: Use 5-10 for publishable results
#
# WHY: Essential for scientific validity. Neural network training is stochastic
#      (random initialization, batch shuffling, dropout). Single-seed results can
#      be misleading due to lucky/unlucky initialization. Multiple seeds enable:
#      1. Statistical significance testing (e.g., Welch's t-test for model comparison)
#      2. Confidence intervals on reported accuracies
#      3. Detection of unstable training dynamics
#      Standard practice in ML research: 5-10 seeds balances statistical power
#      against computational cost (10 seeds = 10x training time).
NUM_SEEDS_DEFAULT: int = 1

# SEED_DEFAULT
# ------------
# WHAT: Base seed value for random number generator initialization.
#
# HOW: Sets the starting point for reproducible pseudo-random sequences. When
#      NUM_SEEDS_DEFAULT > 1, subsequent seeds are SEED_DEFAULT + 1, +2, etc.
#      - Value: 42 (ML community convention, arbitrary but widely adopted)
#      - CLI override: --seed N
#      - Affects: PyTorch RNG, NumPy RNG, Python RNG, CUDA operations
#      - Ensures bit-exact reproducibility across machines when combined with
#        deterministic algorithms (cudnn.deterministic = True)
#
# WHY: Reproducibility is foundational to scientific method. Same seed + same code
#      + same hardware should produce identical results, enabling:
#      1. Bug detection (deviations indicate non-determinism)
#      2. Peer verification (others can replicate exact results)
#      3. Ablation studies (isolate effect of single hyperparameter change)
#      Without fixed seeds, debugging is impossible and results are unreviewable.
SEED_DEFAULT: int = 42

# SEEDS_DEFAULT
# -------------
# WHAT: Pre-computed list of seed values for multi-seed experiments.
#
# HOW: Automatically generated as consecutive integers starting from SEED_DEFAULT:
#      [42, 43, 44, ...] up to NUM_SEEDS_DEFAULT values. Used by main training
#      loop to iterate over independent runs. Do NOT modify directly - instead
#      adjust NUM_SEEDS_DEFAULT and/or SEED_DEFAULT.
#
# WHY: Ensures consistency across separate experiment runs. If different users
#      manually choose seeds, results become incomparable. This list guarantees
#      everyone uses the exact same seeds for apples-to-apples comparison.
#      Pre-computation avoids accidental seed overlap or non-uniform sampling.
SEEDS_DEFAULT: List[int] = list(range(SEED_DEFAULT, SEED_DEFAULT + NUM_SEEDS_DEFAULT))

###################################################################################
# DATASET SIZE CONFIGURATION ######################################################
###################################################################################

# NUM_TRAIN_DEFAULT
# -----------------
# WHAT: Number of samples in the training dataset.
#
# HOW: Controls training set size via stratified random sampling from MNIST's
#      60,000 training images. Must be divisible by 10 for balanced classes.
#      - Range: 1,000 (quick tests) to 60,000 (full MNIST)
#      - CLI override: --num-train N
#      - Default: 30,000 (50% of MNIST, ~18 sec/epoch on RTX 4060)
#      - With data augmentation: Effective dataset size is 4-5x larger
#
# WHY: Training set size directly impacts model generalization and convergence time.
#      Scientific considerations:
#      1. TQF-ANN exploits Z6/D6/T24 symmetries, so it can learn rotation invariance
#         more efficiently than baselines - but still needs sufficient samples to
#         discover these patterns
#      2. With 6x rotation augmentation (0deg, 60deg, 120deg, 180deg, 240deg, 300deg),
#         58K samples provide ~348K augmented views per epoch
#      3. RTX 4060 GPU (8GB VRAM) handles 58K efficiently without memory pressure
#      4. Smaller datasets (10K) risk underfitting; 30K works but 58K yields +1.15%
#         test accuracy improvement by reducing data-limited underfitting
#      Note: MNIST has 60K total training images; 2K are reserved for validation
#      (NUM_VAL_DEFAULT), so 58K is the maximum usable training set.
#      Balance: 58K maximizes available data for 651K-parameter model generalization.
NUM_TRAIN_DEFAULT: int = 58000

# NUM_VAL_DEFAULT
# ---------------
# WHAT: Number of samples in the validation dataset for early stopping.
#
# HOW: Stratified sampling ensures 200 images per digit class (perfectly balanced).
#      Validation set is disjoint from training and test sets.
#      - Default: 2,000 total (200 per class)
#      - CLI override: --num-val N
#      - Used for: Early stopping, learning rate scheduling, hyperparameter tuning
#      - NOT used for: Final model selection or performance reporting
#
# WHY: Prevents overfitting and enables early stopping. Larger validation sets provide:
#      1. More stable loss estimates (reduces noise in stopping criterion)
#      2. Better early stopping decisions (finer resolution for MIN_DELTA)
#      3. More reliable learning rate scheduling (smoother plateau detection)
#      Scientific rationale for 2,000:
#      - Balanced classes (200/class) ensure no digit is overrepresented
#      - Large enough to detect true overfitting vs. random fluctuations
#      - Small enough to leave majority of data for training (60,000 - 30,000 train
#        - 2,000 val - 8,000 test = 20,000 remaining for future experiments)
#      Early stopping prevents wasting compute on plateaued models and improves
#      generalization by avoiding overfitting to training set noise.
NUM_VAL_DEFAULT: int = 2000

# NUM_TEST_UNROT_DEFAULT
# ----------------------
# WHAT: Number of samples in the unrotated (0-degree) test set.
#
# HOW: Stratified sampling from MNIST test set (10,000 images). Tests model
#      performance on canonical MNIST digits (upright, standard orientation).
#      - Default: 8,000 (800 per class)
#      - CLI override: --num-test-unrot N
#      - Augmentation: NONE (pure test distribution)
#
# WHY: Measures generalization to standard MNIST distribution without rotation.
#      This baseline test set answers: "Does the model learn MNIST at all?"
#      Critical for debugging - if unrotated accuracy is low, the model has
#      fundamental issues beyond rotation invariance. Rotated accuracy can NEVER
#      exceed unrotated accuracy (rotation adds difficulty). We use 8,000 to
#      maximize test set size while reserving 2,000 for rotated testing.
NUM_TEST_UNROT_DEFAULT: int = 8000

# NUM_TEST_ROT_DEFAULT
# --------------------
# WHAT: Number of BASE samples in rotated test set (multiplied by 6 rotations).
#
# HOW: Each base sample is rotated to 6 angles: [0deg, 60deg, 120deg, 180deg,
#      240deg, 300deg]. Actual test set size = NUM_TEST_ROT_DEFAULT * 6.
#      - Default: 2,000 base (800 per class after stratification, 12,000 total)
#      - CLI override: --num-test-rot N
#      - Purpose: Tests rotation invariance via exhaustive T24 angle coverage
#
# WHY: Core experimental objective is testing rotation invariance under Z6/D6/T24
#      symmetries. Six rotations (60-degree increments) span the D6 dihedral group.
#      Scientific rationale:
#      1. TQF-ANN's radial dual lattice is designed for 6-fold rotational symmetry
#      2. Testing at 0deg, 60deg, 120deg, etc. verifies Z6 symmetry exploitation
#      3. Smaller base (2,000) keeps computation tractable (2000 * 6 = 12,000 forward
#         passes vs. 8000 * 6 = 48,000 for full test set)
#      Rotation testing is computationally expensive but essential - it's the PRIMARY
#      hypothesis test for whether TQF-ANN's geometric inductive bias improves
#      rotation invariance vs. standard CNNs.
NUM_TEST_ROT_DEFAULT: int = 2000

###################################################################################
# TRAINING HYPERPARAMETERS ########################################################
###################################################################################

# BATCH_SIZE_DEFAULT
# ------------------
# WHAT: Number of samples per mini-batch during training.
#
# HOW: Standard SGD batch size for gradient updates. Larger batches = more stable
#      gradients but less frequent updates. Smaller batches = noisier gradients but
#      more exploration.
#      - Default: 128
#      - CLI override: --batch-size N
#      - Memory constraint: RTX 4060 (8GB VRAM) supports up to 256 for MNIST
#      - Effective batch size with gradient accumulation: 128 (no accumulation used)
#
# WHY: Batch size affects training dynamics and generalization. Scientific considerations:
#      1. Small batches (32-64): High gradient noise, slow convergence, strong regularization
#      2. Medium batches (128-256): Balanced trade-off, industry standard for MNIST
#      3. Large batches (512+): Fast convergence, but can overfit (sharp minima)
#      Empirical findings:
#      - 128 provides stable convergence in ~50 epochs for 30K training samples
#      - Larger batches (256) save ~10% training time but risk overfitting
#      - Smaller batches (64) increase training time by ~20% with minimal accuracy gains
#      Decision: 128 balances efficiency and generalization for fair model comparison.
BATCH_SIZE_DEFAULT: int = 128

# LEARNING_RATE_DEFAULT
# ---------------------
# WHAT: Initial learning rate for Adam optimizer.
#
# HOW: Step size for gradient descent weight updates. Decayed via cosine annealing.
#      - Default: 0.001 (1e-3, standard Adam starting LR)
#      - CLI override: --learning-rate F
#      - Warmup: Linear ramp from 1e-6 to 0.001 over LEARNING_RATE_WARMUP_EPOCHS
#      - Scheduler: Cosine annealing to 1e-6 minimum over remaining epochs
#
# WHY: Learning rate is the most critical hyperparameter in deep learning. Too high
#      causes divergence (loss explodes), too low causes slow convergence (wasted time).
#      Scientific rationale for 0.001:
#      1. Standard Adam default (proven effective across many tasks)
#      2. Warmup prevents early training instability (large gradients in random init)
#      3. Cosine annealing provides smooth decay (avoids step-function discontinuities)
#      Empirical tuning:
#      - 0.01: Too aggressive, training unstable (loss oscillates)
#      - 0.001: Stable convergence in ~50 epochs
#      - 0.0001: Too conservative, requires 200+ epochs for same accuracy
#      Decision: 0.001 with warmup and cosine annealing optimizes training efficiency.
LEARNING_RATE_DEFAULT: float = 0.001

# LEARNING_RATE_WARMUP_EPOCHS
# ---------------------------
# WHAT: Number of epochs for linear learning rate warmup from 1e-6 to LEARNING_RATE_DEFAULT.
#
# HOW: LR starts at 1e-6 and increases linearly to 0.001 over the first 5 epochs.
#      After warmup, cosine annealing scheduler takes over for smooth decay.
#      - Default: 5 epochs
#      - CLI override: --learning-rate-warmup-epochs N
#      - Motivation: Prevents early training instability from large gradients
#
# WHY: Warmup is critical for stable training with large learning rates. At initialization,
#      weights are random and gradients can be extremely large. Applying full LR = 0.001
#      immediately can cause:
#      1. Exploding gradients (loss shoots to infinity)
#      2. Bad local minima (model gets stuck in poor regions)
#      3. Loss oscillations (training never stabilizes)
#      Warmup gradually increases LR, allowing model to "settle" into a good region before
#      aggressive optimization begins. 5 epochs is standard in modern deep learning
#      (e.g., BERT, ViT, ResNet training recipes). Empirical validation shows warmup
#      improves final accuracy by ~0.5-1% and reduces training variance across seeds.
LEARNING_RATE_WARMUP_EPOCHS: int = 5

# WEIGHT_DECAY_DEFAULT
# --------------------
# WHAT: L2 regularization coefficient for Adam optimizer.
#
# HOY: Penalizes large weights to prevent overfitting. Adds 0.5 * weight_decay * ||W||^2
#      to loss function. Equivalent to L2 regularization in traditional ML.
#      - Default: 0.0001 (1e-4)
#      - CLI override: --weight-decay F
#      - Applied to: All weights and biases (PyTorch Adam default behavior)
#      - Interacts with: Dropout, label smoothing (combined regularization)
#
# WHY: Weight decay is a simple but effective regularizer that prevents overfitting by
#      encouraging smaller weight magnitudes. Scientific rationale:
#      1. Large weights amplify noise in input features (overfitting)
#      2. Small weights generalize better (smoother decision boundaries)
#      3. 1e-4 is standard for Adam (equivalent to L2 penalty of 0.0001)
#      Empirical tuning:
#      - 0: No regularization, overfits on 30K samples (val accuracy plateaus early)
#      - 1e-4: Balanced, ~0.5% validation improvement over no decay
#      - 1e-3: Too aggressive, underfits (model capacity constrained)
#      Decision: 1e-4 is widely adopted standard, proven effective for MNIST-scale tasks.
WEIGHT_DECAY_DEFAULT: float = 0.0001

# LABEL_SMOOTHING_DEFAULT
# -----------------------
# WHAT: Label smoothing factor for softmax cross-entropy loss.
#
# HOW: Converts hard labels [0, 0, 1, 0, ...] to soft labels [eps/9, eps/9, 1-eps, ...].
#      Smoothing = 0.1 means true class gets 0.9 probability, others share 0.1.
#      - Default: 0.1 (10% smoothing)
#      - CLI override: --label-smoothing F
#      - Range: [0, 1) where 0 = no smoothing, 0.5 = uniform distribution
#      - Implementation: PyTorch CrossEntropyLoss(label_smoothing=0.1)
#
# WHY: Label smoothing prevents overconfidence and improves generalization. Without
#      smoothing, model is encouraged to output probability 1.0 for correct class,
#      0.0 for all others. This causes:
#      1. Overconfident predictions (poor calibration)
#      2. Overfitting (model memorizes training labels too precisely)
#      3. Brittleness (small input perturbations cause confident wrong predictions)
#      Smoothing encourages model to output probabilities closer to true uncertainty.
#      Empirical validation:
#      - 0.0: Overfits, calibration error increases
#      - 0.1: Standard choice, improves generalization by ~0.3-0.5%
#      - 0.3: Too much smoothing, model becomes underconfident
#      Decision: 0.1 is widely adopted in image classification (e.g., ImageNet training).
LABEL_SMOOTHING_DEFAULT: float = 0.1

# DROPOUT_DEFAULT
# ---------------
# WHAT: Dropout probability for regularization in fully connected layers.
#
# HOW: During training, randomly sets neuron activations to zero with probability p.
#      During inference, all neurons active but outputs scaled by (1-p).
#      - Default: 0.2 (drop 20% of neurons)
#      - CLI override: --dropout F
#      - Applied to: All dense layers except output layer
#      - Combined with: Weight decay, label smoothing (multiple regularizers)
#
# WHY: Dropout is a powerful regularizer that prevents co-adaptation of neurons.
#      Without dropout, neurons can learn to rely on each other's presence, leading
#      to brittle representations that don't generalize. Dropout forces each neuron
#      to be useful independently. Scientific rationale:
#      1. Equivalent to training an ensemble of 2^N networks (exponentially many models)
#      2. Prevents overfitting by adding noise during training
#      3. Improves test accuracy by averaging over many implicit subnetworks
#      Empirical tuning:
#      - 0.0: Overfits on 30K samples, ~1% lower test accuracy
#      - 0.2: Balanced, standard choice for FC layers
#      - 0.5: Too aggressive, underfits (model capacity reduced too much)
#      Decision: 0.2 is standard for image classification (CNNs typically use 0.2-0.3).
DROPOUT_DEFAULT: float = 0.2

# MAX_EPOCHS_DEFAULT
# ------------------
# WHAT: Maximum number of training epochs before stopping.
#
# HOW: One epoch = one pass through entire training dataset. Training stops when:
#      (1) max_epochs reached, OR (2) early stopping triggered by validation plateau.
#      - Default: 100 epochs
#      - CLI override: --num-epochs N
#      - Typical convergence: ~50-80 epochs for MNIST with early stopping
#      - Maximum time: ~1 hour per model on RTX 4060 (18 sec/epoch * 100)
#
# WHY: Maximum epochs prevent infinite training loops and set computational budget.
#      100 epochs is chosen to be generous (2-3x typical convergence time) while
#      remaining computationally tractable. Scientific considerations:
#      1. Early stopping typically triggers around epoch 50-80 (validation plateaus)
#      2. 100 epochs ensure we never prematurely stop due to hard limit
#      3. Prevents wasting compute if early stopping fails (e.g., due to bugs)
#      Empirical observations:
#      - With 58K data, best accuracy epochs are 125-135 (vs 50-80 with 30K)
#      - CosineAnnealingLR fine-tuning tail (epochs 100-150, LR < 3e-4) yields
#        an additional ~0.5% accuracy gain over stopping at 100 epochs
#      - 150 epochs with patience=25 adds +0.22% test acc over 100 epochs
#      Decision: 150 provides headroom for the cosine annealing tail to refine.
MAX_EPOCHS_DEFAULT: int = 150

# PATIENCE_DEFAULT
# ----------------
# WHAT: Number of epochs to wait for validation improvement before early stopping.
#
# HOW: Early stopping monitors validation loss. If validation loss doesn't improve
#      by MIN_DELTA for PATIENCE consecutive epochs, training stops.
#      - Default: 15 epochs
#      - CLI override: --patience N
#      - Saves: Best model checkpoint (lowest validation loss)
#      - Prevents: Overfitting and wasted computation
#
# WHY: Early stopping is essential for efficiency and generalization. Without it:
#      1. Training continues past convergence (wastes time)
#      2. Model overfits to training data (test accuracy decreases)
#      3. No principled stopping criterion (arbitrary epoch count)
#      Patience controls stopping sensitivity. Scientific considerations:
#      - Small patience (5): Stops quickly, risks premature stopping if validation is noisy
#      - Medium patience (15): Balanced, allows temporary plateaus but stops if stuck
#      - Large patience (30): Patient, but wastes compute on plateaued models
#      Empirical tuning:
#      - 10 epochs: Sometimes stops prematurely during learning rate transitions
#      - 15 epochs: Stops too early with 58K data (best epochs at 125-135)
#      - 25 epochs: Allows the CosineAnnealingLR tail to fully converge, yielding
#        +0.22% test accuracy and +0.25% rotated accuracy over patience=15
#      Decision: 25 epochs matches the longer training schedule needed for 58K data.
PATIENCE_DEFAULT: int = 25

# MIN_DELTA_DEFAULT
# -----------------
# WHAT: Minimum validation loss improvement required to reset early stopping counter.
#
# HOW: Validation loss must decrease by at least MIN_DELTA to count as "improvement".
#      Changes smaller than MIN_DELTA are treated as "no improvement" for patience counting.
#      - Default: 0.0005 (5e-4)
#      - CLI override: --min-delta F
#      - Prevents: Early stopping from noise in validation loss
#      - Enables: More stable stopping decisions
#
# WHY: Validation loss is noisy due to finite validation set size. Small random
#      fluctuations (e.g., 0.0001) don't indicate true improvement. MIN_DELTA acts
#      as a noise filter, requiring statistically significant improvements. Scientific
#      rationale:
#      1. Validation loss fluctuates by ~0.001 due to random batch sampling
#      2. 0.0005 threshold filters out >90% of noise-driven fluctuations
#      3. Real improvements are typically >0.001 (clearly above noise floor)
#      Empirical validation:
#      - 0: Too sensitive, early stopping triggered by noise (stops at epoch 30-40)
#      - 0.0005: Balanced, filters noise while catching true plateaus
#      - 0.001: Too strict, ignores incremental improvements (trains unnecessarily long)
#      Decision: 0.0005 is standard practice, effective for cross-entropy loss scale (~0.1-2.0).
MIN_DELTA_DEFAULT: float = 0.0005

# SCHEDULER_T_MAX_DEFAULT
# -----------------------
# WHAT: Period for cosine annealing learning rate scheduler (excluding warmup epochs).
#
# HOW: Computed as MAX_EPOCHS_DEFAULT - LEARNING_RATE_WARMUP_EPOCHS. Defines the
#      number of epochs over which LR decays from LEARNING_RATE_DEFAULT to eta_min=1e-6.
#      - Default: 95 (100 max epochs - 5 warmup epochs)
#      - Automatically adjusted: If --num-epochs changed, T_max updates accordingly
#      - Scheduler formula: lr = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * epoch / T_max))
#
# WHY: Cosine annealing provides smooth learning rate decay without abrupt changes.
#      Unlike step decay (lr *= 0.1 every 30 epochs), cosine decay is continuous and
#      differentiable. Benefits:
#      1. Smooth transitions (no loss spikes when LR drops)
#      2. Exploration-exploitation balance (high LR early, low LR late for fine-tuning)
#      3. Well-studied in literature (used in state-of-the-art training recipes)
#      T_max controls decay speed. Scientific considerations:
#      - T_max = total_epochs: Decay spreads over entire training (smooth, gradual)
#      - T_max < total_epochs: Faster decay, reaches eta_min before training ends
#      - T_max > total_epochs: Slower decay, never reaches eta_min (not recommended)
#      Decision: Set T_max = remaining epochs after warmup for natural decay trajectory.
SCHEDULER_T_MAX_DEFAULT: int = MAX_EPOCHS_DEFAULT - LEARNING_RATE_WARMUP_EPOCHS

###################################################################################
# TQF-SPECIFIC ARCHITECTURE PARAMETERS ############################################
###################################################################################

# TQF_TRUNCATION_R_DEFAULT
# ------------------------
# WHAT: Truncation radius (R) for the radial dual triangular lattice graph.
#
# HOW: Defines maximum graph radius in hyperbolic space. Graph extends from origin
#      to radius R, creating a finite lattice for neural network computation.
#      - Default: 3 (integer hyperbolic distance units)
#      - CLI override: --tqf-R N
#      - Affects: Number of lattice vertices (more vertices = more parameters)
#      - Constraint: Must be > TQF_RADIUS_R_FIXED (inversion radius)
#
# WHY: Truncation radius controls the "size" of the TQF graph structure. Larger R
#      means more lattice points, which decreases parameter density because the
#      number of parameters is fixed (~650K) for fair comparison.
TQF_TRUNCATION_R_DEFAULT: int = 20

# TQF_RADIUS_R_FIXED
# ------------------
# WHAT: Fixed inversion radius (r) for Mobius transformations in hyperbolic space.
#
# HOW: Defines the radius for circle inversion operations used in TQF dual metrics.
#      This is a MATHEMATICAL CONSTANT, not a tunable hyperparameter.
#      - Value: 1.0 (standard hyperbolic geometry)
#      - NOT adjustable: Changing this alters fundamental geometry
#      - Used in: Dual metric computations, hyperbolic geodesic distances
#
# WHY: The inversion radius r = 1.0 is mathematically canonical for the Poincare disk
#      model of hyperbolic geometry. This is not a hyperparameter - it's a geometric
#      definition. Changing r would require redefining all dual metric formulas.
#      Mathematical background:
#      1. Poincare disk: Points at distance < 1 from origin represent hyperbolic space
#      2. Geodesics: Circular arcs perpendicular to unit circle boundary
#      3. Inversion: z -> r^2 / conj(z) maps interior to exterior (duality operation)
#      Setting r = 1.0 is standard convention in hyperbolic geometry literature.
#      Alternative: Could use r != 1, but would require rescaling all distance formulas.
#      Decision: Keep r = 1.0 for mathematical consistency with established theory.
TQF_RADIUS_R_FIXED: float = 1.0

# TQF_HIDDEN_DIMENSION_DEFAULT
# ----------------------------
# WHAT: Hidden feature dimension for TQF graph neural network layers.
#
# HOW: Each lattice vertex has a feature vector of dimension hidden_dim. This is
#      analogous to "number of channels" in CNNs or "hidden size" in Transformers.
#      - Default: 512 (auto-tuned to match ~650K parameters)
#      - CLI override: --tqf-hidden-dim N
#      - Affects: Parameter count (scales quadratically with hidden_dim)
#      - Parameter matching: Auto-adjusted if needed to hit TARGET_PARAMS
#
# WHY: Hidden dimension controls model expressiveness. Larger hidden_dim = more
#      representational capacity but also more parameters. Scientific considerations:
#      1. Small hidden_dim (128): Fast, low capacity, may underfit
#      2. Medium hidden_dim (512): Balanced, standard for ~650K params
#      3. Large hidden_dim (1024): High capacity, requires larger parameter budget
#      For fair comparison with baselines:
#      - All models must have ~650K parameters (+/- 1%)
#      - TQF hidden_dim is auto-tuned to achieve this target
#      - Manual override available for ablation studies
#      Empirical tuning:
#      - hidden_dim = 512 with R = 3.0 yields ~643K params (within tolerance)
#      - Increasing to 768 would exceed budget (~900K params)
#      Decision: 512 provides good expressiveness while matching parameter constraint.
TQF_HIDDEN_DIMENSION_DEFAULT: int = 512

# TQF_SYMMETRY_LEVEL_DEFAULT
# --------------------------
# WHAT: Symmetry group to enforce in TQF architecture.
#
# HOW: Specifies which mathematical symmetry group the network should respect.
#      Options: 'none' (no symmetry), 'Z6' (6-fold rotation), 'D6' (dihedral),
#      'T24' (full triangle group, includes Z6, D6, and reflections).
#      - Default: 'none' (opt-in for symmetry enforcement)
#      - CLI override: --tqf-symmetry-level {none, Z6, D6, T24}
#      - Affects: Network architecture, equivariance constraints, inference cost
#
# WHY: Symmetry groups encode geometric priors for rotation invariance. The TQF
#      framework is designed to exploit these symmetries for better generalization.
#      Mathematical background:
#      1. Z6 (cyclic): 6-fold rotational symmetry (60-degree increments)
#      2. D6 (dihedral): Z6 + reflection symmetry (12 elements total)
#      3. T24 (triangle): Full symmetry group of radial dual lattice (24 elements)
#      Performance characteristics:
#      - 'none': Standard forward pass, fastest inference
#      - 'Z6': 6x inference cost, improves rotated accuracy +1-2%
#      - 'D6': 12x inference cost, adds reflection robustness
#      - 'T24': 24x inference cost, maximum symmetry enforcement
#      Default 'none' provides baseline performance; enable higher symmetry levels
#      via CLI when rotation invariance is specifically required.
TQF_SYMMETRY_LEVEL_DEFAULT: str = 'none'

###################################################################################
# TQF REGULARIZATION WEIGHTS ######################################################
###################################################################################

# TQF_GEOMETRY_REG_WEIGHT_DEFAULT
# -------------------------------
# WHAT: Weight for geometry preservation regularization loss.
#
# HOW: Penalizes deviations from ideal hyperbolic geometry. Added to total loss as:
#      total_loss = classification_loss + geometry_weight * geometry_loss.
#      - Default: 0.0 (disabled, opt-in via CLI)
#      - CLI override: --tqf-geometry-reg-weight F
#      - Recommended range when enabled: 0.001 to 0.01
#
# WHY: TQF's geometric structure (dual metrics, hyperbolic distances) can be
#      preserved during training via this regularization term. Geometry loss measures:
#      1. Triangle inequality violations (non-metric distances)
#      2. Symmetry breaking (loss of Z6/D6 equivariance)
#      3. Hyperbolic property violations (negative curvature)
#      Default 0.0 allows baseline training without geometric constraints.
#      Enable via CLI when geometric preservation is specifically required.
TQF_GEOMETRY_REG_WEIGHT_DEFAULT: float = 0.0

# Note: TQF equivariance/invariance/duality loss weights have been removed from
# config.py as these features are disabled by default. Users enable them by
# providing the weight value directly via CLI (e.g., --tqf-z6-equivariance-weight 0.01).
# See cli.py for valid ranges: Z6 [0.001, 2.0], D6 [0.001, 0.05], T24 [0.001, 0.02], Inversion [0.0, 10.0].


###################################################################################
# TQF DUAL METRICS PARAMETERS #####################################################
###################################################################################

# TQF_DUALITY_TOLERANCE_DEFAULT
# -----------------------------
# WHAT: Tolerance for dual metric consistency verification.
#
# HOW: Maximum allowed relative error in duality equation: |d(x,y) - d_dual(inv(x), inv(y))| / d(x,y).
#      If error exceeds tolerance, emit warning (diagnostic, not a stopping criterion).
#      - Default: 0.01 (1% relative error)
#      - Used in: Dual metric verification checks (every N epochs)
#
# WHY: Duality is a mathematical property that should be preserved during training.
#      Tolerance defines what constitutes "acceptable" numerical error vs. "broken" geometry.
#      Scientific rationale:
#      1. Perfect duality (error = 0) is impossible due to floating-point arithmetic
#      2. Small error (<1%) is acceptable (numerical precision limit)
#      3. Large error (>5%) indicates implementation bugs or training instabilities
#      0.01 (1%) is standard tolerance for geometric computations (IEEE 754 single precision).
#      Decision: 1% allows reasonable numerical error while catching major issues.
TQF_DUALITY_TOLERANCE_DEFAULT: float = 0.01

# TQF_VERIFY_DUALITY_INTERVAL_DEFAULT
# -----------------------------------
# WHAT: Epoch interval for running duality verification checks.
#
# HOW: Every N epochs, compute dual metric consistency and check against DUALITY_TOLERANCE.
#      Emits warnings if duality violations detected (but training continues).
#      - Default: 10 epochs
#      - CLI override: --tqf-verify-duality-interval N
#      - Purpose: Catch geometry degradation early (diagnostic tool)
#
# WHY: Duality verification is computationally expensive (requires full distance matrix
#      computation). Checking every epoch would slow training by ~20%. Checking every
#      10 epochs is a reasonable compromise:
#      1. Frequent enough to catch issues before they compound
#      2. Infrequent enough to have minimal performance impact
#      3. Standard practice: Verification checks typically run every 5-20 epochs
#      If duality violations detected, can enable --tqf-verify-geometry for full diagnostic.
#      Decision: 10 epochs balances diagnostic frequency with computational efficiency.
TQF_VERIFY_DUALITY_INTERVAL_DEFAULT: int = 10

# Z6_DATA_AUGMENTATION_DEFAULT
# -----------------------------
# WHAT: Whether to apply Z6-aligned rotation augmentation during training.
#
# HOW: When True, training images are randomly rotated at 60-degree intervals
#      (with +/-15 degree jitter) to teach rotation robustness.
#      - Default: False (disabled)
#      - CLI override: --z6-data-augmentation (sets to True)
#      - When True: Training with rotation augmentation (improves rotated accuracy)
#      - When False: No rotation augmentation (standard training)
#      - Applies to ALL models (shared training DataLoader), not just TQF-ANN.
#
# WHY: Z6 rotation augmentation is disabled by default because it conflicts with
#      TQF orbit mixing features and can hurt accuracy when both are active.
#      Users can opt in via --z6-data-augmentation when rotation robustness is needed
#      and orbit mixing is not being used.
#      Decision: False by default. Enable explicitly when needed.
Z6_DATA_AUGMENTATION_DEFAULT: bool = False

# TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT
# ---------------------------------------
# WHAT: Temperature for Z6 rotation confidence weighting in orbit mixing.
#
# HOW: Controls sharpness of max-logit weighting across 6 rotation variants.
#      - Default: 0.5 (experimentally optimal for rotation accuracy)
#      - CLI override: --tqf-orbit-mixing-temp-rotation
#      - Range: [0.01, 2.0]
#      - Lower = sharper (best rotation dominates), Higher = more uniform averaging
#
# WHY: Z6 rotations are input-space transformations that produce the most diverse
#      predictions. Experimental results (symmetry_orbit_mark2, Feb 2026) show
#      T=0.5 achieves the best rotation accuracy (67.42%), outperforming both
#      sharper (T=0.3) and softer (T=1.0 → 67.27%, T=2.0 → 65.45%) settings.
#      Decision: 0.5 is the experimentally validated optimum.
TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT: float = 0.5

# TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT
# -----------------------------------------
# WHAT: Temperature for D6 reflection confidence weighting in orbit mixing.
#
# HOW: Controls sharpness of weighting between base and reflected feature variants.
#      - Default: 0.5 (moderate — some digits are asymmetric under reflection)
#      - CLI override: --tqf-orbit-mixing-temp-reflection
#      - Range: [0.01, 2.0]
#
# WHY: Reflections are a weaker symmetry for digit recognition because some digits
#      (e.g., 6/9, 2/5) are not reflection-symmetric. Softer weighting than rotation
#      prevents reflected variants from overriding correct base predictions.
#      Decision: 0.5 provides moderate contribution without overwhelming base.
TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT: float = 0.5

# TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT
# ----------------------------------------
# WHAT: Temperature for T24 zone-swap confidence weighting in orbit mixing.
#
# HOW: Controls sharpness of weighting between normal and zone-swapped variants.
#      - Default: 0.7 (soft — circle inversion is the most abstract symmetry)
#      - CLI override: --tqf-orbit-mixing-temp-inversion
#      - Range: [0.01, 2.0]
#
# WHY: Zone-swap (exchanging inner/outer roles) is the most abstract T24 operation.
#      The model learns specific roles for each zone during training, so swapping
#      them produces less reliable predictions. Soft weighting (high temperature)
#      prevents the zone-swap variant from dominating.
#      Decision: 0.7 gives gentle contribution without diluting primary predictions.
TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT: float = 0.7

# TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT
# -----------------------------------------
# WHAT: Confidence scoring mode for orbit mixing weight computation.
#
# HOW: Controls how each orbit variant's "confidence" is measured before temperature
#      scaling and softmax weighting.
#      - 'max_logit': uses the raw max logit value (original behaviour)
#      - 'margin': uses top-1 minus top-2 logit gap (more discriminative — a variant
#        with a large gap is genuinely confident; a high max with a close second is not)
#      - Default: 'max_logit' (backward-compatible)
#      - CLI override: --tqf-z6-orbit-mixing-confidence-mode {max_logit,margin}
TQF_Z6_ORBIT_MIXING_CONFIDENCE_MODE_DEFAULT: str = 'max_logit'

# TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT
# ------------------------------------------
# WHAT: Aggregation space for combining weighted orbit variant predictions.
#
# HOW: Controls what is averaged across the weighted orbit ensemble.
#      - 'logits': weighted sum of raw logits (original behaviour)
#      - 'probs': weighted sum of softmax probabilities (bounded, no scale blowup)
#      - 'log_probs': weighted sum in log-probability space (geometric mean /
#        product-of-experts — requires broad agreement for high confidence)
#      - Default: 'logits' (backward-compatible)
#      - CLI override: --tqf-z6-orbit-mixing-aggregation-mode {logits,probs,log_probs}
TQF_Z6_ORBIT_MIXING_AGGREGATION_MODE_DEFAULT: str = 'logits'

# TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT
# --------------------------------
# WHAT: Number of highest-confidence orbit variants to include in the ensemble.
#
# HOW: When set, only the top-K variants (by confidence score) are included;
#      the rest are discarded before weighting.
#      - None: use all 6 variants (default, backward-compatible)
#      - 2-6: keep only the top-K most confident variants
#      - CLI override: --tqf-z6-orbit-mixing-top-k INT
TQF_Z6_ORBIT_MIXING_TOP_K_DEFAULT: None = None  # type: ignore[assignment]

# TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT / _ALPHA_DEFAULT
# --------------------------------------------------------
# WHAT: Per-sample adaptive temperature scaling for orbit mixing.
#
# HOW: When enabled, the temperature is scaled per sample based on the orbit
#      entropy (disagreement across variants). High disagreement → softer weighting;
#      low disagreement → sharper weighting.
#      T_eff = T_base * (1 + alpha * orbit_entropy / log(N))
#      - Default: False (fixed temperature, backward-compatible)
#      - CLI override: --tqf-z6-orbit-mixing-adaptive-temp
#      - alpha controls sensitivity; range [0.1, 10.0]. Default: 1.0
TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_DEFAULT: bool = False
TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_DEFAULT: float = 1.0

# TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT
# ----------------------------------------
# WHAT: Interpolation mode for image rotation in orbit mixing.
#
# HOW: Passed to F.grid_sample when rotating input images.
#      - 'bilinear': standard bilinear interpolation (default, backward-compatible)
#      - 'bicubic': higher-quality cubic interpolation; better edge preservation
#      - CLI override: --tqf-z6-orbit-mixing-rotation-mode {bilinear,bicubic}
TQF_Z6_ORBIT_MIXING_ROTATION_MODE_DEFAULT: str = 'bilinear'

# TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT
# -----------------------------------------------
# WHAT: Padding mode for image rotation — controls how corners are filled.
#
# HOW: Passed to F.grid_sample when rotating input images.
#      - 'zeros': fill rotated corners with 0 (default, backward-compatible)
#      - 'border': replicate edge pixels into corners; avoids artificial
#        black-region artefacts that can confuse the model
#      - CLI override: --tqf-z6-orbit-mixing-rotation-padding-mode {zeros,border}
TQF_Z6_ORBIT_MIXING_ROTATION_PADDING_MODE_DEFAULT: str = 'zeros'

# TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT
# ----------------------------------------
# WHAT: Pad the image before rotating, then crop back to 28x28.
#
# HOW: When > 0, the image is padded to (28+2*pad)×(28+2*pad) with reflect
#      padding before rotation, then center-cropped back to 28×28. This prevents
#      any zero-filled corner artefacts from appearing in the rotated output.
#      - 0: disabled (no padding, rotate in place — default, backward-compatible)
#      - 4: typical useful value (adds 4px on each side)
#      - CLI override: --tqf-z6-orbit-mixing-rotation-pad INT
TQF_Z6_ORBIT_MIXING_ROTATION_PAD_DEFAULT: int = 0

# TQF_Z6_NON_ROTATION_AUGMENTATION_DEFAULT
# ----------------------------------------
# WHAT: Enable non-rotation training augmentation (random crop + brightness jitter).
#
# HOW: When True, adds random crop (pad=2, 28×28 crop) and slight brightness /
#      contrast jitter (±10%) to the training transform. Works independently of
#      Z6 rotation augmentation — can be used with or without it.
#      - Default: False (disabled, backward-compatible)
#      - CLI override: --tqf-z6-non-rotation-augmentation
TQF_Z6_NON_ROTATION_AUGMENTATION_DEFAULT: bool = False

# TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_DEFAULT / _ROTATIONS_DEFAULT
# ----------------------------------------------------------
# WHAT: Training-time orbit consistency self-distillation loss.
#
# HOW: During training, run num_rotations additional Z6-rotated forward passes,
#      compute the soft orbit-mixed ensemble, and minimise KL(ensemble || each
#      rotation's softmax). This teaches the model to produce consistent predictions
#      across rotations in output space — different from equivariance losses which
#      operate on intermediate features.
#      - None: disabled (no extra forward passes — default, backward-compatible)
#      - float in [0.0001, 1.0]: weight for the consistency loss term
#      - CLI override: --tqf-z6-orbit-consistency-weight FLOAT
#      - num_rotations: how many Z6 rotations to sample per training batch [1, 5]
#      - CLI override: --tqf-z6-orbit-consistency-rotations INT
#
# WHY: Unlike equivariance loss (which constrains intermediate features), this
#      loss only requires that the final output distribution is consistent across
#      rotations — a strictly weaker and more learnable constraint.
TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_DEFAULT: None = None  # type: ignore[assignment]
TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_DEFAULT: int = 2

###################################################################################
# NUMERIC RANGE CONSTANTS #########################################################
###################################################################################
# Why: Single source of truth for all numeric validation ranges.
#      Used by config.py assertions and cli.py validation/help text.
#      Centralizes all min/max constraints to prevent inconsistencies.

# Training hyperparameter ranges
NUM_SEEDS_MIN: int = 1
NUM_SEEDS_MAX: int = 20
SEED_START_MIN: int = 0
NUM_EPOCHS_MIN: int = 1
NUM_EPOCHS_MAX: int = 200
BATCH_SIZE_MIN: int = 1
BATCH_SIZE_MAX: int = 1024
LEARNING_RATE_MIN: float = 0.0
LEARNING_RATE_MAX: float = 1.0
WEIGHT_DECAY_MIN: float = 0.0
WEIGHT_DECAY_MAX: float = 1.0
LABEL_SMOOTHING_MIN: float = 0.0
LABEL_SMOOTHING_MAX: float = 1.0
PATIENCE_MIN: int = 1
PATIENCE_MAX: int = 50
MIN_DELTA_MIN: float = 0.0
MIN_DELTA_MAX: float = 1.0
LEARNING_RATE_WARMUP_EPOCHS_MIN: int = 0
LEARNING_RATE_WARMUP_EPOCHS_MAX: int = 10

# Dataset size ranges
NUM_TRAIN_MIN: int = 100
NUM_TRAIN_MAX: int = 60000
NUM_VAL_MIN: int = 10
NUM_VAL_MAX: int = 10000
NUM_TEST_ROT_MIN: int = 100
NUM_TEST_ROT_MAX: int = 10000
NUM_TEST_UNROT_MIN: int = 100
NUM_TEST_UNROT_MAX: int = 10000

# TQF architecture ranges
TQF_R_MIN: int = 2  # Must be > inversion radius (r=1)
TQF_R_MAX: int = 100
TQF_HIDDEN_DIM_MIN: int = 8
TQF_HIDDEN_DIM_MAX: int = 512
# TQF orbit mixing temperature ranges
TQF_ORBIT_MIXING_TEMP_MIN: float = 0.01
TQF_ORBIT_MIXING_TEMP_MAX: float = 2.0

# TQF orbit mixing enhancement ranges
TQF_Z6_ORBIT_MIXING_TOP_K_MIN: int = 2
TQF_Z6_ORBIT_MIXING_TOP_K_MAX: int = 6
TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MIN: float = 0.1
TQF_Z6_ORBIT_MIXING_ADAPTIVE_TEMP_ALPHA_MAX: float = 10.0
TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MIN: int = 0
TQF_Z6_ORBIT_MIXING_ROTATION_PAD_MAX: int = 8
TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MIN: float = 0.0001
TQF_Z6_ORBIT_CONSISTENCY_WEIGHT_MAX: float = 1.0
TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MIN: int = 1
TQF_Z6_ORBIT_CONSISTENCY_ROTATIONS_MAX: int = 5

# TQF loss weight ranges
TQF_GEOMETRY_REG_WEIGHT_MIN: float = 0.0
TQF_GEOMETRY_REG_WEIGHT_MAX: float = 10.0
TQF_INVERSION_LOSS_WEIGHT_MIN: float = 0.0
TQF_INVERSION_LOSS_WEIGHT_MAX: float = 10.0
TQF_Z6_EQUIVARIANCE_WEIGHT_MIN: float = 0.001
TQF_Z6_EQUIVARIANCE_WEIGHT_MAX: float = 2.0
TQF_D6_EQUIVARIANCE_WEIGHT_MIN: float = 0.001
TQF_D6_EQUIVARIANCE_WEIGHT_MAX: float = 0.05
TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN: float = 0.001
TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX: float = 0.02

# TQF verification ranges
TQF_VERIFY_DUALITY_INTERVAL_MIN: int = 1
TQF_VERIFY_DUALITY_INTERVAL_MAX: int = 100

###################################################################################
# PARAMETER MATCHING CONFIGURATION ################################################
###################################################################################

# TARGET_PARAMS
# -------------
# WHAT: Target parameter count for all models in "apples-to-apples" comparison.
#
# HOW: All baseline models (MLP, CNN, ResNet) and TQF-ANN are tuned to have
#      approximately this many parameters (+/- TOLERANCE_PERCENT).
#      - Default: 650,000 parameters
#      - Not adjustable via CLI (hardcoded for fair comparison)
#      - Enforced: Training halts if model deviates beyond tolerance
#
# WHY: Fair model comparison requires controlled parameter count. Larger models
#      have more capacity and can achieve higher accuracy simply due to more parameters.
#      Scientific rationale for parameter matching:
#      1. Isolates architectural differences (geometry) from capacity differences (size)
#      2. Industry standard: Compare models at fixed parameter budget (e.g., MobileNet vs ResNet at 5M params)
#      3. Enables meaningful interpretation (if TQF wins, it's due to geometry, not size)
#      Why 650K specifically:
#      - Large enough for meaningful learning on MNIST (sufficient capacity)
#      - Small enough for fast training on RTX 4060 (<30 sec/epoch)
#      - Typical for medium-scale image classification (between 500K-1M params)
#      - Achievable for all architectures (MLP, CNN, ResNet, TQF) without awkward sizes
#      Decision: 650K provides fair comparison point across diverse architectures.
TARGET_PARAMS: int = 650000

# TARGET_PARAMS_TOLERANCE_PERCENT
# -------------------------------
# WHAT: Percentage tolerance for parameter count deviation from TARGET_PARAMS.
#
# HOW: Models must satisfy: |model_params - TARGET_PARAMS| / TARGET_PARAMS <= TOLERANCE / 100.
#      - Default: 1.1% (650K +/- 7,150 params = 642,850 to 657,150)
#      - Verified before training: All models checked, training aborts if violation
#      - Logged at startup: Full parameter breakdown printed for transparency
#
# WHY: Perfect parameter matching is impossible (discrete architectural constraints).
#      Tolerance defines acceptable deviation. Scientific considerations:
#      1. Too tight (<0.5%): Impossible to achieve across diverse architectures
#      2. Balanced (1-2%): Achievable for all models with careful tuning
#      3. Too loose (>5%): Defeats purpose of parameter matching
#      Empirical validation:
#      - 1.1% allows all models (MLP, CNN, ResNet, TQF) to hit target with reasonable
#        hidden dimensions (no awkward sizes like 487 or 1023)
#      - Models at 643K vs 656K params show <0.5% accuracy difference (within statistical noise)
#      - Standard in ML literature: 1-5% parameter tolerance is typical for controlled comparisons
#      Decision: 1.1% is conservative, ensuring tight parameter matching while remaining achievable.
TARGET_PARAMS_TOLERANCE_PERCENT: float = 1.1

# TARGET_PARAMS_TOLERANCE_ABSOLUTE
# --------------------------------
# WHAT: Absolute parameter count deviation allowed (derived from percentage tolerance).
#
# HOW: Computed as: TARGET_PARAMS * TARGET_PARAMS_TOLERANCE_PERCENT / 100
#      - Value: 7,150 params (= 650,000 * 0.011)
#      - Used in verification: abs(model_params - 650000) <= 7150
#
# WHY: Convenience constant for parameter matching verification. Avoids repeated
#      percentage calculations and makes tolerance checks more readable. This is a
#      derived constant, not an independent hyperparameter - changing it requires
#      changing TARGET_PARAMS_TOLERANCE_PERCENT.
TARGET_PARAMS_TOLERANCE_ABSOLUTE: int = int(TARGET_PARAMS * TARGET_PARAMS_TOLERANCE_PERCENT / 100)

###################################################################################
# HARDWARE CONFIGURATION ##########################################################
###################################################################################

# NUM_WORKERS_DEFAULT
# -------------------
# WHAT: Number of parallel worker processes for data loading.
#
# HOW: PyTorch DataLoader spawns this many workers to load and preprocess data in
#      parallel with training. More workers = faster data loading but higher CPU/RAM usage.
#      - Default: 0 (main process only, no multiprocessing)
#      - CLI override: --num-workers N
#      - Platform-dependent: 0 recommended for Windows, 4-8 for Linux/Mac
#
# WHY: Data loading can be bottleneck if CPU preprocessing (augmentation, normalization)
#      is slow relative to GPU training. Parallel workers hide this latency.
#      Scientific rationale for 0 on Windows:
#      1. Windows multiprocessing has issues with PyTorch (CUDA initialization errors)
#      2. MNIST is small (28x28 images), preprocessing is fast (<1ms/image)
#      3. Using workers=0 ensures stable, reproducible training on Windows systems
#      Note: With 58K samples, disk I/O was a severe bottleneck without caching.
#      CustomMNIST now uses lazy in-memory caching (~47 MB for 60K images) so
#      each image is loaded from disk only once, then served from RAM.
#      Recommendation for Linux/Mac: Set to 4-8 if data loading becomes bottleneck.
NUM_WORKERS_DEFAULT: int = 0

# PIN_MEMORY_DEFAULT
# ------------------
# WHAT: Whether to use pinned (page-locked) memory for faster CPU-to-GPU transfers.
#
# HOW: Pinned memory cannot be paged to disk, enabling faster DMA transfers to GPU.
#      PyTorch DataLoader uses this for batch tensors if pin_memory=True.
#      - Default: True
#      - CLI override: --pin-memory / --no-pin-memory
#      - Requirement: CUDA-enabled GPU (automatically disabled on CPU-only systems)
#      - Memory overhead: ~10-20MB for typical batch sizes (negligible)
#
# WHY: Accelerates data transfer from CPU RAM to GPU VRAM. Benefits:
#      1. Faster transfers: 2-3x speedup on CPU->GPU copy (DMA vs paged transfer)
#      2. Overlapping: Transfers can overlap with GPU computation (asynchronous)
#      3. Minimal cost: Small memory overhead, no CPU performance impact
#      Scientific rationale for True:
#      - On RTX 4060 with CUDA, pinned memory reduces data transfer time from ~200ms
#         to ~80ms per epoch (not a huge bottleneck, but free speedup)
#      - Disadvantage: Requires ~20MB pinned memory (negligible on 32GB system RAM)
#      - Automatically disabled if no GPU detected (pin_memory has no effect on CPU)
#      Recommendation: Always enable on CUDA systems (default), disable only if encountering
#      memory pressure (very rare).
PIN_MEMORY_DEFAULT: bool = True

###################################################################################
# SNN-SPECIFIC PARAMETERS (Reserved for TQF-SNN Extension) ########################
###################################################################################

# TIMESTEPS
# ---------
# WHAT: Number of discrete timesteps for Spiking Neural Network (SNN) simulations.
#
# HOW: SNNs process input as temporal sequences of spikes over T timesteps. Each timestep
#      represents 1ms of biological time. Higher T = more temporal resolution but slower.
#      - Default: 100
#      - Future CLI override: --snn-timesteps N
#      - Typical range: 50 (fast, coarse) to 200 (slow, fine)
#
# WHY: Temporal dynamics are core to SNNs (unlike ANNs which are feedforward). Number
#      of timesteps controls:
#      1. Temporal resolution: Finer timesteps (200) capture faster spike patterns
#      2. Information capacity: Longer sequences encode more information (rate coding)
#      3. Biological plausibility: ~100-200ms typical for human visual processing
#      Scientific rationale for 100:
#      - MNIST is static image, so temporal dynamics are artificial (input encoded as
#         constant rates over 100 timesteps)
#      - 100 timesteps is standard in SNN literature for static image tasks
#      - Fewer timesteps (50): May not converge, loses accuracy
#      - More timesteps (200): Marginal gains (<1%), 2x computational cost
#      This parameter is reserved for the TQF-SNN extension and is not used in the
#      current TQF-ANN implementation.
TIMESTEPS: int = 100

###################################################################################
# VALIDATION AND CONSISTENCY CHECKS ###############################################
###################################################################################

# Dataset size assertions
assert NUM_TRAIN_MIN <= NUM_TRAIN_DEFAULT <= NUM_TRAIN_MAX, \
    f"NUM_TRAIN_DEFAULT must be in [{NUM_TRAIN_MIN}, {NUM_TRAIN_MAX}]"
assert NUM_VAL_MIN <= NUM_VAL_DEFAULT <= NUM_VAL_MAX, \
    f"NUM_VAL_DEFAULT must be in [{NUM_VAL_MIN}, {NUM_VAL_MAX}]"
assert NUM_TEST_ROT_MIN <= NUM_TEST_ROT_DEFAULT <= NUM_TEST_ROT_MAX, \
    f"NUM_TEST_ROT_DEFAULT must be in [{NUM_TEST_ROT_MIN}, {NUM_TEST_ROT_MAX}]"
assert NUM_TEST_UNROT_MIN <= NUM_TEST_UNROT_DEFAULT <= NUM_TEST_UNROT_MAX, \
    f"NUM_TEST_UNROT_DEFAULT must be in [{NUM_TEST_UNROT_MIN}, {NUM_TEST_UNROT_MAX}]"

# Training hyperparameter assertions
assert LEARNING_RATE_MIN < LEARNING_RATE_DEFAULT <= LEARNING_RATE_MAX, \
    f"Learning rate must be in ({LEARNING_RATE_MIN}, {LEARNING_RATE_MAX}]"
assert 0 <= DROPOUT_DEFAULT < 1, "Dropout must be in [0, 1)"
assert LABEL_SMOOTHING_MIN <= LABEL_SMOOTHING_DEFAULT <= LABEL_SMOOTHING_MAX, \
    f"Label smoothing must be in [{LABEL_SMOOTHING_MIN}, {LABEL_SMOOTHING_MAX}]"
assert PATIENCE_MIN <= PATIENCE_DEFAULT <= PATIENCE_MAX, \
    f"Patience must be in [{PATIENCE_MIN}, {PATIENCE_MAX}]"
assert MIN_DELTA_MIN <= MIN_DELTA_DEFAULT <= MIN_DELTA_MAX, \
    f"Min delta must be in [{MIN_DELTA_MIN}, {MIN_DELTA_MAX}]"
assert LEARNING_RATE_WARMUP_EPOCHS_MIN <= LEARNING_RATE_WARMUP_EPOCHS <= LEARNING_RATE_WARMUP_EPOCHS_MAX, \
    f"Warmup epochs must be in [{LEARNING_RATE_WARMUP_EPOCHS_MIN}, {LEARNING_RATE_WARMUP_EPOCHS_MAX}]"
assert NUM_EPOCHS_MIN <= MAX_EPOCHS_DEFAULT <= NUM_EPOCHS_MAX, \
    f"Max epochs must be in [{NUM_EPOCHS_MIN}, {NUM_EPOCHS_MAX}]"
assert BATCH_SIZE_MIN <= BATCH_SIZE_DEFAULT <= BATCH_SIZE_MAX, \
    f"Batch size must be in [{BATCH_SIZE_MIN}, {BATCH_SIZE_MAX}]"
assert WEIGHT_DECAY_MIN <= WEIGHT_DECAY_DEFAULT <= WEIGHT_DECAY_MAX, \
    f"Weight decay must be in [{WEIGHT_DECAY_MIN}, {WEIGHT_DECAY_MAX}]"

# TQF parameter assertions
assert TQF_R_MIN <= TQF_TRUNCATION_R_DEFAULT <= TQF_R_MAX, \
    f"Truncation radius must be in [{TQF_R_MIN}, {TQF_R_MAX}]"
assert TQF_TRUNCATION_R_DEFAULT > TQF_RADIUS_R_FIXED, \
    "Truncation radius must exceed inversion radius"
assert TQF_HIDDEN_DIM_MIN <= TQF_HIDDEN_DIMENSION_DEFAULT <= TQF_HIDDEN_DIM_MAX, \
    f"Hidden dimension must be in [{TQF_HIDDEN_DIM_MIN}, {TQF_HIDDEN_DIM_MAX}]"
assert TQF_SYMMETRY_LEVEL_DEFAULT in ['none', 'Z6', 'D6', 'T24'], \
    "Invalid symmetry level"
# Regularization weight assertions
assert TQF_GEOMETRY_REG_WEIGHT_MIN <= TQF_GEOMETRY_REG_WEIGHT_DEFAULT <= TQF_GEOMETRY_REG_WEIGHT_MAX, \
    f"Geometry reg weight must be in [{TQF_GEOMETRY_REG_WEIGHT_MIN}, {TQF_GEOMETRY_REG_WEIGHT_MAX}]"

# Orbit mixing temperature assertions
assert TQF_ORBIT_MIXING_TEMP_MIN <= TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT <= TQF_ORBIT_MIXING_TEMP_MAX, \
    f"Rotation temperature must be in [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]"
assert TQF_ORBIT_MIXING_TEMP_MIN <= TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT <= TQF_ORBIT_MIXING_TEMP_MAX, \
    f"Reflection temperature must be in [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]"
assert TQF_ORBIT_MIXING_TEMP_MIN <= TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT <= TQF_ORBIT_MIXING_TEMP_MAX, \
    f"Inversion temperature must be in [{TQF_ORBIT_MIXING_TEMP_MIN}, {TQF_ORBIT_MIXING_TEMP_MAX}]"

# Note: Equivariance/invariance/duality loss weight assertions removed - these features
# are disabled by default and weights are validated in cli.py when provided.

# Parameter matching assertions
assert TARGET_PARAMS > 0, "Target params must be positive"
assert 0 < TARGET_PARAMS_TOLERANCE_PERCENT < 100, "Tolerance must be in (0, 100)"
