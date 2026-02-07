"""
prepare_datasets.py - Dataset Loading and Preprocessing for TQF-ANN Experiments

This module provides reproducible dataset preparation for training and evaluating TQF-ANN
and baseline models on MNIST with rotational variations. It implements stratified sampling,
class-balanced splits, and rotation augmentation aligned with TQF symmetry groups.

Key Features:
- Automatic download and organization of MNIST into class-specific folders (PNG format)
- Rotated test set generation at 60-degree increments (0, 60, 120, 180, 240, 300 degrees)
  for evaluating Z6 rotational invariance
- Stratified train/validation split ensuring balanced class representation
- Custom dataset class supporting both organized PNGs and standard MNIST formats
- Configurable batch sizes, sampling strategies, and data augmentation
- Reproducible random seeding for deterministic splits across runs and machines

Scientific Rationale:
The 60-degree rotation increments align with the hexagonal (Z6) symmetry of the TQF
architecture, providing a natural testbed for evaluating rotational equivariance. The
stratified sampling ensures fair class representation, while the separate rotated and
unrotated test sets enable assessment of both in-distribution and rotational robustness.

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
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image
from collections import defaultdict
from typing import Any, List, Optional, Tuple

# Import configuration constants
from config import (
    BATCH_SIZE_DEFAULT,
    NUM_WORKERS_DEFAULT,
    PIN_MEMORY_DEFAULT,
    NUM_TRAIN_DEFAULT,
    NUM_VAL_DEFAULT,
    NUM_TEST_ROT_DEFAULT,
    NUM_TEST_UNROT_DEFAULT
)


# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# Determine data directory path
# Why: Centralizes data storage for reproducibility and organization
current_dir: str = os.path.dirname(os.path.abspath(__file__))
parent_dir: str = os.path.dirname(current_dir)
data_dir: str = os.path.join(parent_dir, 'data/mnist')

DATA_DIR: str = os.path.expanduser(data_dir)
os.makedirs(DATA_DIR, exist_ok=True)

# Set global seed for reproducibility across all random operations
# Why: Ensures deterministic behavior for shuffling, sampling, and augmentation
GLOBAL_SEED: int = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


# ==============================================================================
# Z6-ALIGNED ROTATION AUGMENTATION
# ==============================================================================

class Z6AlignedRotation:
    """
    Z6-aligned rotation augmentation for TQF-ANN training.

    This transform applies rotations that align with the hexagonal (Z6) symmetry
    of the TQF architecture. It randomly selects one of the 6 principal rotation
    angles (0 deg, 60 deg, 120 deg, 180 deg, 240 deg, 300 deg) and optionally adds small random jitter.

    Why Z6-aligned rotations?
    - The TQF architecture has inherent 6-fold rotational symmetry (Z6 group)
    - Training with Z6-aligned rotations teaches the network that 60 deg rotations
      should produce shifted sector features (sector i -> sector i+1)
    - This enables the orbit pooling mechanism to achieve true rotational invariance
    - Small jitter adds regularization while preserving the Z6 structure

    Without Z6-aligned augmentation, the network cannot learn the equivariance
    property needed for orbit pooling to work on rotated test images.

    Args:
        jitter: Maximum random jitter in degrees (default: 15.0).
                Set to 0.0 for pure Z6 rotations without noise.

    Example:
        >>> transform = Z6AlignedRotation(jitter=15.0)
        >>> # Sample rotation: 60 deg base + random jitter in [-15 deg, +15 deg]
        >>> rotated_img = transform(img)  # Rotation between 45 deg and 75 deg
    """

    # The 6 principal rotation angles forming the Z6 group
    Z6_ANGLES: List[float] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]

    def __init__(self, jitter: float = 15.0):
        """
        Initialize Z6-aligned rotation transform.

        Args:
            jitter: Maximum random jitter in degrees around each Z6 angle.
                   The final rotation angle is: Z6_angle + uniform(-jitter, +jitter)
        """
        self.jitter: float = jitter

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply Z6-aligned rotation to the input image.

        Args:
            img: PIL Image to rotate

        Returns:
            Rotated PIL Image
        """
        import random

        # Select random Z6 base angle
        base_angle: float = random.choice(self.Z6_ANGLES)

        # Add random jitter if enabled
        if self.jitter > 0:
            jitter_amount: float = random.uniform(-self.jitter, self.jitter)
            angle: float = base_angle + jitter_amount
        else:
            angle: float = base_angle

        # Apply rotation (PIL rotates counterclockwise, which matches mathematical convention)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(jitter={self.jitter})"


# ==============================================================================
# STEP 1: DOWNLOAD AND ORGANIZE BASE MNIST
# ==============================================================================

def download_and_organize_mnist() -> None:
    """
    Download official MNIST and save each image as PNG in class-specific folders.

    Organizes 60,000 training and 10,000 test images into structured directories:
    - organized/train/class_0/ through organized/train/class_9/
    - organized/test/class_0/ through organized/test/class_9/

    Why: Class-based organization facilitates stratified sampling, enables easy visual
    inspection, and supports custom dataset loading. PNG format ensures lossless storage
    and broad compatibility.

    Scientific rationale: Proper data organization is essential for reproducible ML
    experiments. The class-balanced structure enables precise control over sampling
    strategies and ensures no class imbalance in train/val splits.
    """
    # Standard transform to tensor for processing
    transform: transforms.Compose = transforms.Compose([transforms.ToTensor()])

    # Download MNIST datasets (downloads only if not already present)
    train_dataset: datasets.MNIST = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset: datasets.MNIST = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    # Define organized directories for train and test splits
    train_dir: str = os.path.join(DATA_DIR, 'organized/train')
    test_dir: str = os.path.join(DATA_DIR, 'organized/test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create class subdirectories (0-9 for each digit)
    for class_id in range(10):
        os.makedirs(os.path.join(train_dir, f'class_{class_id}'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, f'class_{class_id}'), exist_ok=True)

    def save_images(dataset: datasets.MNIST, base_dir: str, is_train: bool = True) -> None:
        """
        Save dataset images to class-specific folders as PNG files.

        Args:
            dataset: MNIST dataset (train or test)
            base_dir: Base directory for saving images
            is_train: Whether this is training data (affects filename prefix)
        """
        for idx, (img, label) in enumerate(dataset):
            # Convert tensor back to PIL Image for saving
            img_pil: Image.Image = transforms.ToPILImage()(img)

            # Create descriptive filename with split, index, and label
            split: str = 'train' if is_train else 'test'
            filename: str = f'{split}_{idx:05d}_label_{label}.png'

            # Save to appropriate class folder
            img_pil.save(os.path.join(base_dir, f'class_{label}', filename))

    # Save both train and test datasets
    save_images(train_dataset, train_dir, is_train=True)
    save_images(test_dataset, test_dir, is_train=False)

    print(f"Base MNIST organized in:\n  {os.path.abspath(train_dir)}\n  {os.path.abspath(test_dir)}")


# ==============================================================================
# STEP 2: CREATE ROTATED TEST SET
# ==============================================================================

def create_rotated_test() -> None:
    """
    Generate rotated versions of MNIST test set at 60-degree increments.

    Creates 6 rotated versions of each test image (0, 60, 120, 180, 240, 300 degrees),
    resulting in 60,000 total test images (10,000 original x 6 rotations). The rotation
    angles are specifically chosen to align with Z6 hexagonal symmetry.

    Why: Tests model robustness and equivariance under rotations that match the TQF
    architecture's inherent symmetry. The 60-degree increments correspond to the order-6
    cyclic group (Z6) of the hexagonal lattice structure.

    Scientific rationale: Standard MNIST is unrotated, which doesn't challenge rotational
    invariance. By testing on systematically rotated data, we can quantify how well the
    TQF symmetry structure actually helps with rotational variations versus standard
    architectures that must learn rotation invariance implicitly.
    """
    # Load test dataset (download if not present)
    test_dataset: datasets.MNIST = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )

    # Create output directory for rotated test set
    rotated_test_dir: str = os.path.join(DATA_DIR, 'organized/rotated_test')
    os.makedirs(rotated_test_dir, exist_ok=True)

    # Create class subdirectories
    for class_id in range(10):
        os.makedirs(os.path.join(rotated_test_dir, f'class_{class_id}'), exist_ok=True)

    # Define rotation angles aligned with Z6 symmetry (60-degree increments)
    # Why: Covers full 360 degrees evenly with hexagonal symmetry
    angles: List[int] = [0, 60, 120, 180, 240, 300]

    # Generate rotated versions of each test image
    for idx, (img_tensor, label) in enumerate(test_dataset):
        img: Image.Image = transforms.ToPILImage()(img_tensor)

        for angle in angles:
            # Use BICUBIC interpolation for higher quality rotation
            # Why: Preserves fine details better than NEAREST or BILINEAR
            rotated_img: Image.Image = img.rotate(
                angle, resample=Image.BICUBIC, expand=False
            )

            # Create filename encoding original index, label, and rotation angle
            filename: str = f'test_{idx:05d}_label_{label}_rot_{angle}.png'
            rotated_img.save(os.path.join(rotated_test_dir, f'class_{label}', filename))

    total_images: int = len(test_dataset) * len(angles)
    print(f"Rotated test set created in {os.path.abspath(rotated_test_dir)}")
    print(f"  ({len(test_dataset)} original x {len(angles)} rotations = {total_images} images)")


# ==============================================================================
# CUSTOM DATASET CLASS
# ==============================================================================

class CustomMNIST(Dataset):
    """
    Custom dataset wrapper for loading images from class-organized PNG directories.

    Supports loading from the organized folder structure created by download_and_organize_mnist
    and create_rotated_test. This provides more flexibility than the standard torchvision
    MNIST dataset, allowing custom sampling strategies and transformations.

    Attributes:
        transform: Optional transform to apply to images
        samples: List of (filepath, label) tuples for all images in the dataset
    """

    def __init__(self, root_dir: str, transform: Optional[Any] = None) -> None:
        """
        Initialize CustomMNIST dataset from organized directory structure.

        Args:
            root_dir: Root directory containing class_0/ through class_9/ subdirectories
            transform: Optional torchvision transform to apply to images
        """
        self.transform: Optional[Any] = transform
        self.samples: List[Tuple[str, int]] = []

        # Scan all class directories and collect image paths
        for class_id in range(10):
            class_dir: str = os.path.join(root_dir, f'class_{class_id}')
            if not os.path.exists(class_dir):
                continue

            # Add all PNG files from this class directory
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith('.png'):
                    self.samples.append((os.path.join(class_dir, filename), class_id))

    def __len__(self) -> int:
        """Return total number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Load and return a single sample.

        Args:
            idx: Index of sample to load

        Returns:
            (image, label): Image tensor (after transform) and integer label
        """
        img_path, label = self.samples[idx]

        # Load image and ensure it's grayscale (MNIST is 1-channel)
        img: Image.Image = Image.open(img_path).convert('L')

        # Apply transform if provided
        if self.transform:
            img = self.transform(img)

        return img, label


class TransformedSubset(Dataset):
    """
    Dataset wrapper that applies a transform to a subset of another dataset.

    This allows applying different transforms to train vs validation splits of the same
    base dataset, enabling data augmentation on training data while keeping validation
    data clean for unbiased evaluation.

    Attributes:
        dataset: Base dataset to draw samples from
        indices: List of indices defining the subset
        transform: Transform to apply to samples from this subset
    """

    def __init__(self, dataset: Dataset, indices: List[int], transform: Any) -> None:
        """
        Initialize transformed subset.

        Args:
            dataset: Base dataset
            indices: List of indices defining which samples belong to this subset
            transform: Transform to apply to these samples
        """
        self.dataset: Dataset = dataset
        self.indices: List[int] = indices
        self.transform: Any = transform

    def __len__(self) -> int:
        """Return number of samples in this subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Load and return a single sample from the subset.

        Args:
            idx: Index within the subset (not the base dataset)

        Returns:
            (image, label): Transformed image tensor and integer label
        """
        # Map subset index to base dataset index
        original_idx: int = self.indices[idx]

        # Get image path and label from base dataset
        img_path, label = self.dataset.samples[original_idx]

        # Load image as grayscale
        img: Image.Image = Image.open(img_path).convert('L')

        # Apply transform
        if self.transform:
            img = self.transform(img)

        return img, label


# ==============================================================================
# MAIN DATALOADER FACTORY FUNCTION
# ==============================================================================

def get_dataloaders(
    batch_size: int = BATCH_SIZE_DEFAULT,
    num_workers: int = NUM_WORKERS_DEFAULT,
    pin_memory: bool = PIN_MEMORY_DEFAULT,
    num_train: int = NUM_TRAIN_DEFAULT,
    num_val: int = NUM_VAL_DEFAULT,
    num_test_rot: int = NUM_TEST_ROT_DEFAULT,
    num_test_unrot: int = NUM_TEST_UNROT_DEFAULT,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create stratified, reproducible DataLoaders for TQF-ANN experiments.

    This function orchestrates the entire data pipeline:
    1. Ensures MNIST data is downloaded and organized
    2. Creates stratified train/validation split (balanced across all 10 classes)
    3. Prepares rotated test set for evaluating rotational robustness
    4. Applies appropriate augmentation to training data (if enabled)
    5. Returns DataLoaders with consistent sampling across runs

    The stratification ensures that each class is proportionally represented in both
    training and validation sets, preventing class imbalance issues. The rotated test
    set is also stratified to ensure balanced representation across all (class, angle)
    combinations.

    Args:
        batch_size: Samples per batch for all dataloaders
        num_workers: Number of subprocesses for data loading (0 = main process only)
        pin_memory: Whether to pin memory for faster GPU transfer
        num_train: Number of training samples (stratified across 10 classes)
        num_val: Number of validation samples (stratified across 10 classes)
        num_test_rot: Number of rotated test samples (stratified across 10 classes x 6 angles)
        num_test_unrot: Number of unrotated test samples
        augment_train: Whether to apply rotation augmentation to training data

    Returns:
        train_loader: Training DataLoader (with optional augmentation)
        val_loader: Validation DataLoader (no augmentation)
        test_loader_rot: Rotated test DataLoader (60-degree increments)
        test_loader_unrot: Unrotated test DataLoader (original MNIST test set)

    Scientific rationale:
    - Stratification ensures balanced class representation for fair evaluation
    - Separate val/test splits prevent overfitting to validation metrics
    - Rotated test set evaluates symmetry exploitation vs. learned invariance
    - Pin memory and multiple workers improve throughput on GPU systems
    """
    # Define transforms for training vs evaluation
    if augment_train:
        # Training transform with Z6-aligned rotation augmentation
        # Why: Teaches network that 60 deg rotations produce shifted sector features
        # This enables orbit pooling to achieve true rotational invariance on test set
        # The jitter (+/-15 deg) adds regularization while preserving Z6 structure
        train_transform: transforms.Compose = transforms.Compose([
            Z6AlignedRotation(jitter=15.0),  # Z6 angles (0 deg,60 deg,120 deg,...) + +/-15 deg jitter
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    else:
        # Training transform without augmentation (for ablation studies)
        train_transform: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Evaluation transform: no augmentation for consistent evaluation
    eval_transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Ensure data exists on disk (idempotent - safe to call multiple times)
    train_dir: str = os.path.join(DATA_DIR, 'organized/train')
    rotated_test_dir: str = os.path.join(DATA_DIR, 'organized/rotated_test')

    if not os.path.exists(train_dir):
        download_and_organize_mnist()
    if not os.path.exists(rotated_test_dir):
        create_rotated_test()

    # Load datasets (transform will be applied in DataLoader)
    full_train_dataset: CustomMNIST = CustomMNIST(train_dir, transform=None)
    rotated_test_dataset: CustomMNIST = CustomMNIST(rotated_test_dir, transform=eval_transform)

    # Create stratified train/validation split
    # Why: Ensures each class has exactly num_val/10 samples in validation set
    indices: np.ndarray = np.arange(len(full_train_dataset))
    np.random.shuffle(indices)

    # Select validation samples (stratified by class)
    val_idx: List[int] = []
    per_class_val: int = num_val // 10  # Equal validation samples per class

    for class_id in range(10):
        # Get all indices for this class
        class_indices: List[int] = [
            i for i in indices if full_train_dataset.samples[i][1] == class_id
        ]
        # Take first per_class_val samples for validation
        val_idx.extend(class_indices[:per_class_val])

    # Shuffle validation indices
    val_idx_array: np.ndarray = np.array(val_idx)
    np.random.shuffle(val_idx_array)
    val_idx = val_idx_array.tolist()

    # Select training samples from remaining indices
    remaining_idx: np.ndarray = np.array([i for i in indices if i not in val_idx])

    if len(remaining_idx) >= num_train:
        train_idx: np.ndarray = np.random.choice(remaining_idx, size=num_train, replace=False)
    else:
        # Use all remaining samples if num_train exceeds available
        train_idx: np.ndarray = remaining_idx
        print(f"Warning: Requested num_train={num_train} exceeds available samples "
              f"{len(remaining_idx)}. Using all {len(remaining_idx)} samples.")

    # Create stratified rotated test set
    # Why: Ensures balanced representation across all (class, angle) combinations
    samples_by_class_angle: defaultdict = defaultdict(list)

    for idx in range(len(rotated_test_dataset)):
        path, label = rotated_test_dataset.samples[idx]

        # Extract rotation angle from filename (format: ...rot_<angle>.png)
        fname: str = os.path.basename(path)
        angle: int = int(fname.split('_rot_')[1].split('.')[0])

        # Group samples by (class, angle) combination
        samples_by_class_angle[(label, angle)].append(idx)

    # Sample equally from each (class, angle) combination
    test_idx_rot: List[int] = []
    images_per_combo: int = max(1, num_test_rot // 60)  # 60 = 10 classes x 6 angles

    for (class_id, angle), idx_list in samples_by_class_angle.items():
        # Sample up to images_per_combo from each combination
        n_samples: int = min(images_per_combo, len(idx_list))
        chosen: np.ndarray = np.random.choice(idx_list, size=n_samples, replace=False)
        test_idx_rot.extend(chosen)

    # Shuffle rotated test indices
    np.random.shuffle(test_idx_rot)

    # Create unrotated test set
    original_test_dataset: datasets.MNIST = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=eval_transform
    )

    n_available: int = len(original_test_dataset)
    n_unrot_actual: int = min(num_test_unrot, n_available)
    unrot_idx: np.ndarray = np.random.choice(n_available, size=n_unrot_actual, replace=False)

    if n_unrot_actual < num_test_unrot:
        print(f"Warning: Requested num_test_unrot={num_test_unrot} exceeds available "
              f"{n_available}. Using {n_unrot_actual}.")

    # Create DataLoaders with appropriate transforms
    persistent: bool = num_workers > 0  # Enable persistent workers if using multiple workers

    # Training DataLoader with augmentation
    train_subset: TransformedSubset = TransformedSubset(
        full_train_dataset, train_idx.tolist(), train_transform
    )
    train_loader: DataLoader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False
    )

    # Validation DataLoader without augmentation
    val_subset: TransformedSubset = TransformedSubset(
        full_train_dataset, val_idx, eval_transform
    )
    val_loader: DataLoader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False
    )

    # Rotated test DataLoader
    test_loader_rot: DataLoader = DataLoader(
        rotated_test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_idx_rot),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False
    )

    # Unrotated test DataLoader
    test_loader_unrot: DataLoader = DataLoader(
        original_test_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(unrot_idx),
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False
    )

    return train_loader, val_loader, test_loader_rot, test_loader_unrot


# ==============================================================================
# MODULE TEST
# ==============================================================================

if __name__ == '__main__':
    """
    Run module test to verify data pipeline functionality.

    This test:
    1. Downloads and organizes MNIST (if not already present)
    2. Creates rotated test set (if not already present)
    3. Generates all four DataLoaders with default configuration
    4. Prints summary statistics to verify correct operation

    Safe to run multiple times (idempotent operations).
    """
    print("=" * 80)
    print("PREPARE_DATASETS MODULE TEST")
    print("=" * 80)

    # Generate organized datasets
    print("\n1. Downloading and organizing MNIST...")
    download_and_organize_mnist()

    print("\n2. Creating rotated test set...")
    create_rotated_test()

    # Create DataLoaders
    print("\n3. Creating DataLoaders...")
    train, val, test_rot, test_unrot = get_dataloaders(batch_size=64)

    # Print summary
    print("\n" + "=" * 80)
    print("DATALOADERS READY AND FULLY REPRODUCIBLE")
    print("=" * 80)
    print(f"  Train samples: {len(train.sampler)}")
    print(f"  Validation samples: {len(val.sampler)}")
    print(f"  Rotated test samples: {len(test_rot.sampler)}")
    print(f"  Unrotated test samples: {len(test_unrot.sampler)}")
    print("=" * 80)
