"""
test_config.py - Comprehensive Configuration Tests for TQF-NN

This module tests all configuration parameters in config.py, ensuring they have
valid types, values, and consistency with each other.

Key Test Coverage:
- Reproducibility constants (seeds, NUM_SEEDS_DEFAULT)
- Dataset size configuration (train/val/test splits)
- Training hyperparameters (learning rate, batch size, regularization)
- TQF architecture parameters (lattice geometry, symmetry groups)
- Parameter matching constraints (fair comparison)
- Hardware configuration (CUDA, data loading)
- Cross-parameter consistency checks

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

import unittest
import sys
from typing import List

# Import shared utilities
from conftest import (
    assert_valid_probability,
    assert_positive_integer,
    assert_non_negative
)

# Import config module
try:
    import config
except ImportError as e:
    raise ImportError(f"Cannot import config module: {e}")


class TestReproducibilityConstants(unittest.TestCase):
    """Test suite for reproducibility-related configuration constants."""

    def test_seed_default_is_valid(self) -> None:
        """Test that SEED_DEFAULT is a valid non-negative integer."""
        assert_positive_integer(config.SEED_DEFAULT, "SEED_DEFAULT")
        self.assertEqual(config.SEED_DEFAULT, 42, "Expected default seed of 42")

    def test_seed_default_is_integer(self) -> None:
        """Test that SEED_DEFAULT is of type int."""
        self.assertIsInstance(config.SEED_DEFAULT, int)

    def test_seed_default_non_negative(self) -> None:
        """Test that SEED_DEFAULT is non-negative."""
        self.assertGreaterEqual(config.SEED_DEFAULT, 0)

    def test_num_seeds_default_is_valid(self) -> None:
        """Test that NUM_SEEDS_DEFAULT is a positive integer."""
        assert_positive_integer(config.NUM_SEEDS_DEFAULT, "NUM_SEEDS_DEFAULT")

    def test_num_seeds_default_reasonable_range(self) -> None:
        """Test that NUM_SEEDS_DEFAULT is in reasonable range [1, 20]."""
        self.assertGreaterEqual(config.NUM_SEEDS_DEFAULT, 1)
        self.assertLessEqual(config.NUM_SEEDS_DEFAULT, 20)

    def test_seeds_default_list_valid(self) -> None:
        """Test that SEEDS_DEFAULT list is correctly structured."""
        self.assertEqual(len(config.SEEDS_DEFAULT), config.NUM_SEEDS_DEFAULT)

        expected: List[int] = list(range(config.SEED_DEFAULT,
                                         config.SEED_DEFAULT + config.NUM_SEEDS_DEFAULT))
        self.assertEqual(config.SEEDS_DEFAULT, expected)

        self.assertEqual(len(config.SEEDS_DEFAULT), len(set(config.SEEDS_DEFAULT)))
        self.assertTrue(all(isinstance(s, int) for s in config.SEEDS_DEFAULT))

    def test_seeds_default_all_unique(self) -> None:
        """Test that all seeds in SEEDS_DEFAULT are unique."""
        self.assertEqual(len(config.SEEDS_DEFAULT), len(set(config.SEEDS_DEFAULT)))

    def test_seeds_default_consecutive(self) -> None:
        """Test that SEEDS_DEFAULT contains consecutive integers."""
        for i in range(1, len(config.SEEDS_DEFAULT)):
            self.assertEqual(config.SEEDS_DEFAULT[i], config.SEEDS_DEFAULT[i-1] + 1)


class TestDatasetSizeConfiguration(unittest.TestCase):
    """Test suite for dataset size configuration parameters."""

    def test_dataset_sizes_positive(self) -> None:
        """Test that all dataset sizes are positive integers."""
        for param in ['NUM_TRAIN_DEFAULT', 'NUM_VAL_DEFAULT',
                      'NUM_TEST_ROT_DEFAULT', 'NUM_TEST_UNROT_DEFAULT']:
            value: int = getattr(config, param)
            assert_positive_integer(value, param)

    def test_train_val_divisible_by_10(self) -> None:
        """Test train/val sizes divisible by 10 for MNIST (10 classes)."""
        self.assertEqual(config.NUM_TRAIN_DEFAULT % 10, 0,
                        "NUM_TRAIN_DEFAULT must be divisible by 10")
        self.assertEqual(config.NUM_VAL_DEFAULT % 10, 0,
                        "NUM_VAL_DEFAULT must be divisible by 10")

    def test_dataset_sizes_realistic(self) -> None:
        """Test that dataset sizes don't exceed MNIST limits."""
        self.assertLessEqual(config.NUM_TRAIN_DEFAULT, 60000)
        self.assertLessEqual(config.NUM_VAL_DEFAULT, 10000)
        self.assertLessEqual(config.NUM_TEST_ROT_DEFAULT, 10000)
        self.assertLessEqual(config.NUM_TEST_UNROT_DEFAULT, 10000)

    def test_num_train_type(self) -> None:
        """Test NUM_TRAIN_DEFAULT is integer."""
        self.assertIsInstance(config.NUM_TRAIN_DEFAULT, int)

    def test_num_val_type(self) -> None:
        """Test NUM_VAL_DEFAULT is integer."""
        self.assertIsInstance(config.NUM_VAL_DEFAULT, int)

    def test_num_test_rot_type(self) -> None:
        """Test NUM_TEST_ROT_DEFAULT is integer."""
        self.assertIsInstance(config.NUM_TEST_ROT_DEFAULT, int)

    def test_num_test_unrot_type(self) -> None:
        """Test NUM_TEST_UNROT_DEFAULT is integer."""
        self.assertIsInstance(config.NUM_TEST_UNROT_DEFAULT, int)

    def test_train_greater_than_val(self) -> None:
        """Test that training set is larger than validation set."""
        self.assertGreater(config.NUM_TRAIN_DEFAULT, config.NUM_VAL_DEFAULT)

    def test_test_sets_non_zero(self) -> None:
        """Test that test sets have at least some samples."""
        self.assertGreater(config.NUM_TEST_ROT_DEFAULT, 0)
        self.assertGreater(config.NUM_TEST_UNROT_DEFAULT, 0)


class TestTQFArchitectureParameters(unittest.TestCase):
    """Test suite for TQF-specific architecture parameters."""

    def test_tqf_dims_positive(self) -> None:
        """Test that TQF dimension parameters are positive."""
        float_params: List[str] = ['TQF_RADIUS_R_FIXED']
        int_or_float_params: List[str] = ['TQF_TRUNCATION_R_DEFAULT']  # Can be int or float
        int_params: List[str] = ['TQF_HIDDEN_DIMENSION_DEFAULT']
        # Note: TQF_FRACTAL_ITERATIONS_DEFAULT excluded - can be 0 (disabled by default, opt-in)

        for param in float_params:
            value: float = getattr(config, param)
            self.assertIsInstance(value, float, f"{param} must be float")
            self.assertGreater(value, 0.0, f"{param} must be positive")

        for param in int_or_float_params:
            value = getattr(config, param)
            self.assertIsInstance(value, (int, float), f"{param} must be int or float")
            self.assertGreater(value, 0, f"{param} must be positive")

        for param in int_params:
            value: int = getattr(config, param)
            assert_positive_integer(value, param)

    def test_truncation_greater_than_radius(self) -> None:
        """Test that truncation radius exceeds base radius."""
        self.assertGreater(config.TQF_TRUNCATION_R_DEFAULT, config.TQF_RADIUS_R_FIXED)

    def test_symmetry_level_valid(self) -> None:
        """Test that symmetry level is valid."""
        valid: List[str] = ['none', 'Z6', 'D6', 'T24']
        self.assertIn(config.TQF_SYMMETRY_LEVEL_DEFAULT, valid)

    def test_fibonacci_mode_valid(self) -> None:
        """Test that Fibonacci dimension mode is valid."""
        valid: List[str] = ['none', 'linear', 'fibonacci']
        self.assertIn(config.TQF_FIBONACCI_DIMENSION_MODE_DEFAULT, valid)

    def test_tqf_weights_non_negative(self) -> None:
        """Test that TQF weight parameters are non-negative."""
        for param in ['TQF_FRACTAL_DIM_TOLERANCE_DEFAULT',
                     'TQF_SELF_SIMILARITY_WEIGHT_DEFAULT',
                     'TQF_BOX_COUNTING_WEIGHT_DEFAULT']:
            value: float = getattr(config, param)
            assert_non_negative(value, param)

    def test_tqf_radius_type(self) -> None:
        """Test TQF_RADIUS_R_FIXED is float."""
        self.assertIsInstance(config.TQF_RADIUS_R_FIXED, float)

    def test_tqf_truncation_type(self) -> None:
        """Test TQF_TRUNCATION_R_DEFAULT is numeric."""
        self.assertIsInstance(config.TQF_TRUNCATION_R_DEFAULT, (int, float))

    def test_tqf_hidden_dim_type(self) -> None:
        """Test TQF_HIDDEN_DIMENSION_DEFAULT is integer."""
        self.assertIsInstance(config.TQF_HIDDEN_DIMENSION_DEFAULT, int)

    def test_tqf_fractal_iterations_type(self) -> None:
        """Test TQF_FRACTAL_ITERATIONS_DEFAULT is integer."""
        self.assertIsInstance(config.TQF_FRACTAL_ITERATIONS_DEFAULT, int)

    def test_tqf_hidden_dim_reasonable(self) -> None:
        """Test TQF hidden dimension is in reasonable range."""
        self.assertGreaterEqual(config.TQF_HIDDEN_DIMENSION_DEFAULT, 32)
        self.assertLessEqual(config.TQF_HIDDEN_DIMENSION_DEFAULT, 2048)

    def test_tqf_fractal_iterations_reasonable(self) -> None:
        """Test fractal iterations default is 0 (disabled) or in reasonable range [1, 20].

        Fractal iterations is an opt-in feature:
        - 0 = disabled by default
        - [1, 20] = valid enabled range when user provides value via CLI
        """
        self.assertGreaterEqual(config.TQF_FRACTAL_ITERATIONS_DEFAULT, 0)
        self.assertLessEqual(config.TQF_FRACTAL_ITERATIONS_DEFAULT, 20)


class TestTrainingHyperparameters(unittest.TestCase):
    """Test suite for training hyperparameters."""

    def test_hyperparameters_positive(self) -> None:
        """Test that core hyperparameters are positive."""
        for param in ['MAX_EPOCHS_DEFAULT', 'PATIENCE_DEFAULT', 'BATCH_SIZE_DEFAULT']:
            value: int = getattr(config, param)
            assert_positive_integer(value, param)

    def test_probabilities_valid(self) -> None:
        """Test that probability parameters are in [0, 1)."""
        for param in ['LEARNING_RATE_DEFAULT', 'WEIGHT_DECAY_DEFAULT',
                     'DROPOUT_DEFAULT', 'LABEL_SMOOTHING_DEFAULT']:
            value: float = getattr(config, param)
            assert_valid_probability(value, param)

    def test_min_delta_non_negative(self) -> None:
        """Test that MIN_DELTA is non-negative."""
        assert_non_negative(config.MIN_DELTA_DEFAULT, "MIN_DELTA_DEFAULT")

    def test_max_epochs_type(self) -> None:
        """Test MAX_EPOCHS_DEFAULT is integer."""
        self.assertIsInstance(config.MAX_EPOCHS_DEFAULT, int)

    def test_patience_type(self) -> None:
        """Test PATIENCE_DEFAULT is integer."""
        self.assertIsInstance(config.PATIENCE_DEFAULT, int)

    def test_batch_size_type(self) -> None:
        """Test BATCH_SIZE_DEFAULT is integer."""
        self.assertIsInstance(config.BATCH_SIZE_DEFAULT, int)

    def test_learning_rate_type(self) -> None:
        """Test LEARNING_RATE_DEFAULT is float."""
        self.assertIsInstance(config.LEARNING_RATE_DEFAULT, float)

    def test_weight_decay_type(self) -> None:
        """Test WEIGHT_DECAY_DEFAULT is float."""
        self.assertIsInstance(config.WEIGHT_DECAY_DEFAULT, float)

    def test_dropout_type(self) -> None:
        """Test DROPOUT_DEFAULT is float."""
        self.assertIsInstance(config.DROPOUT_DEFAULT, float)

    def test_label_smoothing_type(self) -> None:
        """Test LABEL_SMOOTHING_DEFAULT is float."""
        self.assertIsInstance(config.LABEL_SMOOTHING_DEFAULT, float)

    def test_min_delta_type(self) -> None:
        """Test MIN_DELTA_DEFAULT is float."""
        self.assertIsInstance(config.MIN_DELTA_DEFAULT, float)

    def test_max_epochs_reasonable(self) -> None:
        """Test MAX_EPOCHS in reasonable range [1, 1000]."""
        self.assertGreaterEqual(config.MAX_EPOCHS_DEFAULT, 1)
        self.assertLessEqual(config.MAX_EPOCHS_DEFAULT, 1000)

    def test_patience_reasonable(self) -> None:
        """Test PATIENCE in reasonable range [1, 50]."""
        self.assertGreaterEqual(config.PATIENCE_DEFAULT, 1)
        self.assertLessEqual(config.PATIENCE_DEFAULT, 50)

    def test_batch_size_power_of_two(self) -> None:
        """Test if batch size is power of 2 (recommended for GPU)."""
        batch_size: int = config.BATCH_SIZE_DEFAULT
        # Check if power of 2: (n & (n-1)) == 0 for powers of 2
        is_power_of_2: bool = (batch_size & (batch_size - 1)) == 0 and batch_size > 0
        self.assertTrue(is_power_of_2, f"Batch size {batch_size} should be power of 2")

    def test_learning_rate_reasonable(self) -> None:
        """Test learning rate in typical range [1e-5, 1e-1]."""
        self.assertGreaterEqual(config.LEARNING_RATE_DEFAULT, 1e-5)
        self.assertLessEqual(config.LEARNING_RATE_DEFAULT, 1e-1)


class TestHardwareConfiguration(unittest.TestCase):
    """Test suite for hardware configuration."""

    def test_num_workers_non_negative(self) -> None:
        """Test that NUM_WORKERS_DEFAULT is non-negative."""
        self.assertIsInstance(config.NUM_WORKERS_DEFAULT, int)
        self.assertGreaterEqual(config.NUM_WORKERS_DEFAULT, 0)

    def test_pin_memory_boolean(self) -> None:
        """Test that PIN_MEMORY_DEFAULT is boolean."""
        self.assertIsInstance(config.PIN_MEMORY_DEFAULT, bool)

    def test_num_workers_reasonable(self) -> None:
        """Test NUM_WORKERS in reasonable range [0, 16]."""
        self.assertLessEqual(config.NUM_WORKERS_DEFAULT, 16)


class TestParameterMatchingConfiguration(unittest.TestCase):
    """Test suite for parameter matching configuration."""

    def test_target_params_positive(self) -> None:
        """Test that TARGET_PARAMS is positive."""
        assert_positive_integer(config.TARGET_PARAMS, "TARGET_PARAMS")

    def test_target_params_tolerance_reasonable(self) -> None:
        """Test that parameter tolerance is reasonable [0.1, 50]%."""
        tolerance: float = config.TARGET_PARAMS_TOLERANCE_PERCENT
        self.assertIsInstance(tolerance, (int, float))
        self.assertGreaterEqual(tolerance, 0.1)
        self.assertLessEqual(tolerance, 50.0)

    def test_target_params_tolerance_type(self) -> None:
        """Test TARGET_PARAMS_TOLERANCE_PERCENT is numeric."""
        self.assertIsInstance(config.TARGET_PARAMS_TOLERANCE_PERCENT, (int, float))

    def test_target_params_tolerance_absolute_consistent(self) -> None:
        """Test that absolute tolerance matches percentage tolerance."""
        expected: int = int(config.TARGET_PARAMS * config.TARGET_PARAMS_TOLERANCE_PERCENT / 100)
        self.assertEqual(config.TARGET_PARAMS_TOLERANCE_ABSOLUTE, expected)


class TestConfigurationConsistency(unittest.TestCase):
    """Test suite for cross-parameter consistency."""

    def test_warmup_less_than_max_epochs(self) -> None:
        """Test that LR warmup completes before training ends."""
        if hasattr(config, 'LEARNING_RATE_WARMUP_EPOCHS'):
            self.assertLessEqual(config.LEARNING_RATE_WARMUP_EPOCHS,
                               config.MAX_EPOCHS_DEFAULT)

    def test_scheduler_t_max_positive(self) -> None:
        """Test that scheduler T_max is positive if defined."""
        if hasattr(config, 'SCHEDULER_T_MAX_DEFAULT'):
            assert_positive_integer(config.SCHEDULER_T_MAX_DEFAULT,
                                  "SCHEDULER_T_MAX_DEFAULT")

    def test_scheduler_t_max_consistent(self) -> None:
        """Test that T_max equals max_epochs minus warmup."""
        if hasattr(config, 'SCHEDULER_T_MAX_DEFAULT') and hasattr(config, 'LEARNING_RATE_WARMUP_EPOCHS'):
            expected: int = config.MAX_EPOCHS_DEFAULT - config.LEARNING_RATE_WARMUP_EPOCHS
            self.assertEqual(config.SCHEDULER_T_MAX_DEFAULT, expected)

    def test_patience_less_than_max_epochs(self) -> None:
        """Test that patience is less than max epochs."""
        self.assertLess(config.PATIENCE_DEFAULT, config.MAX_EPOCHS_DEFAULT)

    def test_total_dataset_size_valid(self) -> None:
        """Test that val set fits within MNIST training set.

        NUM_TRAIN_DEFAULT may exceed available samples (e.g. 60000 requested
        when only 58000 remain after validation split). The training code
        handles this gracefully by capping to available samples and warning.
        We only verify that the validation set itself fits within MNIST.
        """
        self.assertLessEqual(config.NUM_VAL_DEFAULT, 60000,
                           "Validation set exceeds MNIST training set")


class TestSNNParameters(unittest.TestCase):
    """Test suite for SNN-specific parameters (future implementation)."""

    def test_timesteps_exists(self) -> None:
        """Test that TIMESTEPS constant exists."""
        self.assertTrue(hasattr(config, 'TIMESTEPS'))

    def test_timesteps_positive(self) -> None:
        """Test that TIMESTEPS is positive."""
        assert_positive_integer(config.TIMESTEPS, "TIMESTEPS")

    def test_timesteps_reasonable(self) -> None:
        """Test that TIMESTEPS is in reasonable range [10, 500]."""
        self.assertGreaterEqual(config.TIMESTEPS, 10)
        self.assertLessEqual(config.TIMESTEPS, 500)


class TestRangeConstants(unittest.TestCase):
    """Test suite for numeric range constants (MIN/MAX bounds).

    WHY: All CLI parameters have min/max range constants defined in config.py
         as the single source of truth. These tests verify the constants exist,
         have correct types, and satisfy MIN < MAX.
    HOW: Check each range constant pair for existence, type, and ordering.
    WHAT: Every MIN constant must be strictly less than its MAX counterpart.
    """

    def test_training_hyperparameter_ranges_exist(self) -> None:
        """Test that all training hyperparameter range constants exist."""
        pairs = [
            ('NUM_SEEDS_MIN', 'NUM_SEEDS_MAX'),
            ('NUM_EPOCHS_MIN', 'NUM_EPOCHS_MAX'),
            ('BATCH_SIZE_MIN', 'BATCH_SIZE_MAX'),
            ('LEARNING_RATE_MIN', 'LEARNING_RATE_MAX'),
            ('WEIGHT_DECAY_MIN', 'WEIGHT_DECAY_MAX'),
            ('LABEL_SMOOTHING_MIN', 'LABEL_SMOOTHING_MAX'),
            ('PATIENCE_MIN', 'PATIENCE_MAX'),
            ('MIN_DELTA_MIN', 'MIN_DELTA_MAX'),
            ('LEARNING_RATE_WARMUP_EPOCHS_MIN', 'LEARNING_RATE_WARMUP_EPOCHS_MAX'),
        ]
        for min_name, max_name in pairs:
            self.assertTrue(hasattr(config, min_name), f"Missing {min_name}")
            self.assertTrue(hasattr(config, max_name), f"Missing {max_name}")

    def test_dataset_size_ranges_exist(self) -> None:
        """Test that all dataset size range constants exist."""
        pairs = [
            ('NUM_TRAIN_MIN', 'NUM_TRAIN_MAX'),
            ('NUM_VAL_MIN', 'NUM_VAL_MAX'),
            ('NUM_TEST_ROT_MIN', 'NUM_TEST_ROT_MAX'),
            ('NUM_TEST_UNROT_MIN', 'NUM_TEST_UNROT_MAX'),
        ]
        for min_name, max_name in pairs:
            self.assertTrue(hasattr(config, min_name), f"Missing {min_name}")
            self.assertTrue(hasattr(config, max_name), f"Missing {max_name}")

    def test_tqf_architecture_ranges_exist(self) -> None:
        """Test that all TQF architecture range constants exist."""
        pairs = [
            ('TQF_R_MIN', 'TQF_R_MAX'),
            ('TQF_HIDDEN_DIM_MIN', 'TQF_HIDDEN_DIM_MAX'),
            ('TQF_FRACTAL_ITERATIONS_MIN', 'TQF_FRACTAL_ITERATIONS_MAX'),
            # TQF_FRACTAL_DIM_TOLERANCE range constants removed (internal, not CLI-tunable)
            ('TQF_SELF_SIMILARITY_WEIGHT_MIN', 'TQF_SELF_SIMILARITY_WEIGHT_MAX'),
            ('TQF_BOX_COUNTING_WEIGHT_MIN', 'TQF_BOX_COUNTING_WEIGHT_MAX'),
            # TQF_BOX_COUNTING_SCALES range constants removed (internal, not CLI-tunable)
            ('TQF_HOP_ATTENTION_TEMP_MIN', 'TQF_HOP_ATTENTION_TEMP_MAX'),
        ]
        for min_name, max_name in pairs:
            self.assertTrue(hasattr(config, min_name), f"Missing {min_name}")
            self.assertTrue(hasattr(config, max_name), f"Missing {max_name}")

    def test_tqf_loss_weight_ranges_exist(self) -> None:
        """Test that all TQF loss weight range constants exist."""
        pairs = [
            ('TQF_GEOMETRY_REG_WEIGHT_MIN', 'TQF_GEOMETRY_REG_WEIGHT_MAX'),
            ('TQF_INVERSION_LOSS_WEIGHT_MIN', 'TQF_INVERSION_LOSS_WEIGHT_MAX'),
            ('TQF_Z6_EQUIVARIANCE_WEIGHT_MIN', 'TQF_Z6_EQUIVARIANCE_WEIGHT_MAX'),
            ('TQF_D6_EQUIVARIANCE_WEIGHT_MIN', 'TQF_D6_EQUIVARIANCE_WEIGHT_MAX'),
            ('TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN', 'TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX'),
        ]
        for min_name, max_name in pairs:
            self.assertTrue(hasattr(config, min_name), f"Missing {min_name}")
            self.assertTrue(hasattr(config, max_name), f"Missing {max_name}")

    def test_orbit_mixing_temp_ranges_exist(self) -> None:
        """Test that orbit mixing temperature range constants exist."""
        self.assertTrue(hasattr(config, 'TQF_ORBIT_MIXING_TEMP_MIN'))
        self.assertTrue(hasattr(config, 'TQF_ORBIT_MIXING_TEMP_MAX'))

    def test_verification_ranges_exist(self) -> None:
        """Test that verification range constants exist."""
        self.assertTrue(hasattr(config, 'TQF_VERIFY_DUALITY_INTERVAL_MIN'))
        self.assertTrue(hasattr(config, 'TQF_VERIFY_DUALITY_INTERVAL_MAX'))

    def test_all_min_less_than_max(self) -> None:
        """Test that MIN < MAX for all range constant pairs.

        WHY: A range where MIN >= MAX is always a bug.
        HOW: Iterate over all known pairs and verify strict ordering.
        WHAT: Every MIN must be strictly less than its MAX.
        """
        pairs = [
            ('NUM_SEEDS_MIN', 'NUM_SEEDS_MAX'),
            ('NUM_EPOCHS_MIN', 'NUM_EPOCHS_MAX'),
            ('BATCH_SIZE_MIN', 'BATCH_SIZE_MAX'),
            ('LEARNING_RATE_MIN', 'LEARNING_RATE_MAX'),
            ('WEIGHT_DECAY_MIN', 'WEIGHT_DECAY_MAX'),
            ('LABEL_SMOOTHING_MIN', 'LABEL_SMOOTHING_MAX'),
            ('PATIENCE_MIN', 'PATIENCE_MAX'),
            ('MIN_DELTA_MIN', 'MIN_DELTA_MAX'),
            ('LEARNING_RATE_WARMUP_EPOCHS_MIN', 'LEARNING_RATE_WARMUP_EPOCHS_MAX'),
            ('NUM_TRAIN_MIN', 'NUM_TRAIN_MAX'),
            ('NUM_VAL_MIN', 'NUM_VAL_MAX'),
            ('NUM_TEST_ROT_MIN', 'NUM_TEST_ROT_MAX'),
            ('NUM_TEST_UNROT_MIN', 'NUM_TEST_UNROT_MAX'),
            ('TQF_R_MIN', 'TQF_R_MAX'),
            ('TQF_HIDDEN_DIM_MIN', 'TQF_HIDDEN_DIM_MAX'),
            ('TQF_FRACTAL_ITERATIONS_MIN', 'TQF_FRACTAL_ITERATIONS_MAX'),
            # TQF_FRACTAL_DIM_TOLERANCE range constants removed (internal, not CLI-tunable)
            ('TQF_SELF_SIMILARITY_WEIGHT_MIN', 'TQF_SELF_SIMILARITY_WEIGHT_MAX'),
            ('TQF_BOX_COUNTING_WEIGHT_MIN', 'TQF_BOX_COUNTING_WEIGHT_MAX'),
            # TQF_BOX_COUNTING_SCALES range constants removed (internal, not CLI-tunable)
            ('TQF_HOP_ATTENTION_TEMP_MIN', 'TQF_HOP_ATTENTION_TEMP_MAX'),
            ('TQF_ORBIT_MIXING_TEMP_MIN', 'TQF_ORBIT_MIXING_TEMP_MAX'),
            ('TQF_GEOMETRY_REG_WEIGHT_MIN', 'TQF_GEOMETRY_REG_WEIGHT_MAX'),
            ('TQF_INVERSION_LOSS_WEIGHT_MIN', 'TQF_INVERSION_LOSS_WEIGHT_MAX'),
            ('TQF_Z6_EQUIVARIANCE_WEIGHT_MIN', 'TQF_Z6_EQUIVARIANCE_WEIGHT_MAX'),
            ('TQF_D6_EQUIVARIANCE_WEIGHT_MIN', 'TQF_D6_EQUIVARIANCE_WEIGHT_MAX'),
            ('TQF_T24_ORBIT_INVARIANCE_WEIGHT_MIN', 'TQF_T24_ORBIT_INVARIANCE_WEIGHT_MAX'),
            ('TQF_VERIFY_DUALITY_INTERVAL_MIN', 'TQF_VERIFY_DUALITY_INTERVAL_MAX'),
        ]
        for min_name, max_name in pairs:
            min_val = getattr(config, min_name)
            max_val = getattr(config, max_name)
            self.assertLess(min_val, max_val,
                           f"{min_name} ({min_val}) must be < {max_name} ({max_val})")

    def test_seed_start_min_non_negative(self) -> None:
        """Test that SEED_START_MIN is non-negative."""
        self.assertTrue(hasattr(config, 'SEED_START_MIN'))
        self.assertGreaterEqual(config.SEED_START_MIN, 0)


class TestDefaultsWithinRanges(unittest.TestCase):
    """Test suite verifying all default values fall within their range constants.

    WHY: If a default value is outside its MIN/MAX range, the config.py
         assertion will catch it at import time. These tests provide explicit
         verification and clearer error messages for debugging.
    HOW: For each default, verify MIN <= DEFAULT <= MAX.
    WHAT: Every default must be within its range bounds.
    """

    def test_training_defaults_within_ranges(self) -> None:
        """Test training hyperparameter defaults are within bounds."""
        checks = [
            ('NUM_SEEDS_MIN', 'NUM_SEEDS_DEFAULT', 'NUM_SEEDS_MAX'),
            ('NUM_EPOCHS_MIN', 'MAX_EPOCHS_DEFAULT', 'NUM_EPOCHS_MAX'),
            ('BATCH_SIZE_MIN', 'BATCH_SIZE_DEFAULT', 'BATCH_SIZE_MAX'),
            ('PATIENCE_MIN', 'PATIENCE_DEFAULT', 'PATIENCE_MAX'),
            ('MIN_DELTA_MIN', 'MIN_DELTA_DEFAULT', 'MIN_DELTA_MAX'),
            ('WEIGHT_DECAY_MIN', 'WEIGHT_DECAY_DEFAULT', 'WEIGHT_DECAY_MAX'),
            ('LABEL_SMOOTHING_MIN', 'LABEL_SMOOTHING_DEFAULT', 'LABEL_SMOOTHING_MAX'),
            ('LEARNING_RATE_WARMUP_EPOCHS_MIN', 'LEARNING_RATE_WARMUP_EPOCHS', 'LEARNING_RATE_WARMUP_EPOCHS_MAX'),
        ]
        for min_name, default_name, max_name in checks:
            min_val = getattr(config, min_name)
            default_val = getattr(config, default_name)
            max_val = getattr(config, max_name)
            self.assertGreaterEqual(default_val, min_val,
                f"{default_name} ({default_val}) must be >= {min_name} ({min_val})")
            self.assertLessEqual(default_val, max_val,
                f"{default_name} ({default_val}) must be <= {max_name} ({max_val})")

    def test_learning_rate_within_range(self) -> None:
        """Test learning rate is within (LEARNING_RATE_MIN, LEARNING_RATE_MAX].

        Note: Learning rate uses strict > for MIN because LR=0.0 is invalid.
        """
        self.assertGreater(config.LEARNING_RATE_DEFAULT, config.LEARNING_RATE_MIN)
        self.assertLessEqual(config.LEARNING_RATE_DEFAULT, config.LEARNING_RATE_MAX)

    def test_dataset_defaults_within_ranges(self) -> None:
        """Test dataset size defaults are within bounds."""
        checks = [
            ('NUM_TRAIN_MIN', 'NUM_TRAIN_DEFAULT', 'NUM_TRAIN_MAX'),
            ('NUM_VAL_MIN', 'NUM_VAL_DEFAULT', 'NUM_VAL_MAX'),
            ('NUM_TEST_ROT_MIN', 'NUM_TEST_ROT_DEFAULT', 'NUM_TEST_ROT_MAX'),
            ('NUM_TEST_UNROT_MIN', 'NUM_TEST_UNROT_DEFAULT', 'NUM_TEST_UNROT_MAX'),
        ]
        for min_name, default_name, max_name in checks:
            min_val = getattr(config, min_name)
            default_val = getattr(config, default_name)
            max_val = getattr(config, max_name)
            self.assertGreaterEqual(default_val, min_val,
                f"{default_name} ({default_val}) must be >= {min_name} ({min_val})")
            self.assertLessEqual(default_val, max_val,
                f"{default_name} ({default_val}) must be <= {max_name} ({max_val})")

    def test_tqf_architecture_defaults_within_ranges(self) -> None:
        """Test TQF architecture defaults are within bounds."""
        self.assertGreaterEqual(config.TQF_TRUNCATION_R_DEFAULT, config.TQF_R_MIN)
        self.assertLessEqual(config.TQF_TRUNCATION_R_DEFAULT, config.TQF_R_MAX)

        self.assertGreaterEqual(config.TQF_HIDDEN_DIMENSION_DEFAULT, config.TQF_HIDDEN_DIM_MIN)
        self.assertLessEqual(config.TQF_HIDDEN_DIMENSION_DEFAULT, config.TQF_HIDDEN_DIM_MAX)

        # TQF_BOX_COUNTING_SCALES range check removed (internal constant, not CLI-tunable)
        # Validated by assertion in config.py: 2 <= TQF_BOX_COUNTING_SCALES_DEFAULT <= 20

        self.assertGreaterEqual(config.TQF_HOP_ATTENTION_TEMP_DEFAULT, config.TQF_HOP_ATTENTION_TEMP_MIN)
        self.assertLessEqual(config.TQF_HOP_ATTENTION_TEMP_DEFAULT, config.TQF_HOP_ATTENTION_TEMP_MAX)

    def test_tqf_weight_defaults_within_ranges(self) -> None:
        """Test TQF regularization weight defaults are within bounds."""
        checks = [
            ('TQF_GEOMETRY_REG_WEIGHT_MIN', 'TQF_GEOMETRY_REG_WEIGHT_DEFAULT', 'TQF_GEOMETRY_REG_WEIGHT_MAX'),
            ('TQF_SELF_SIMILARITY_WEIGHT_MIN', 'TQF_SELF_SIMILARITY_WEIGHT_DEFAULT', 'TQF_SELF_SIMILARITY_WEIGHT_MAX'),
            ('TQF_BOX_COUNTING_WEIGHT_MIN', 'TQF_BOX_COUNTING_WEIGHT_DEFAULT', 'TQF_BOX_COUNTING_WEIGHT_MAX'),
            # TQF_FRACTAL_DIM_TOLERANCE range check removed (internal constant, not CLI-tunable)
        ]
        for min_name, default_name, max_name in checks:
            min_val = getattr(config, min_name)
            default_val = getattr(config, default_name)
            max_val = getattr(config, max_name)
            self.assertGreaterEqual(default_val, min_val,
                f"{default_name} ({default_val}) must be >= {min_name} ({min_val})")
            self.assertLessEqual(default_val, max_val,
                f"{default_name} ({default_val}) must be <= {max_name} ({max_val})")

    def test_orbit_mixing_temp_defaults_within_ranges(self) -> None:
        """Test orbit mixing temperature defaults are within bounds."""
        for default_name in ['TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT',
                             'TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT',
                             'TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT']:
            default_val: float = getattr(config, default_name)
            self.assertGreaterEqual(default_val, config.TQF_ORBIT_MIXING_TEMP_MIN,
                f"{default_name} ({default_val}) must be >= TQF_ORBIT_MIXING_TEMP_MIN ({config.TQF_ORBIT_MIXING_TEMP_MIN})")
            self.assertLessEqual(default_val, config.TQF_ORBIT_MIXING_TEMP_MAX,
                f"{default_name} ({default_val}) must be <= TQF_ORBIT_MIXING_TEMP_MAX ({config.TQF_ORBIT_MIXING_TEMP_MAX})")


class TestOrbitMixingAndAugmentationDefaults(unittest.TestCase):
    """Test suite for orbit mixing and Z6 augmentation configuration defaults.

    WHY: Orbit mixing (evaluation-time ensemble) and Z6 augmentation (training-time
         rotation) are key TQF-ANN features. Their defaults must be valid.
    HOW: Verify existence, type, and value of each default.
    WHAT: Defaults should match the mark 3 spec temperatures and augmentation on.
    """

    def test_z6_augmentation_default_exists(self) -> None:
        """Test that TQF_USE_Z6_AUGMENTATION_DEFAULT exists and is True."""
        self.assertTrue(hasattr(config, 'TQF_USE_Z6_AUGMENTATION_DEFAULT'))
        self.assertIsInstance(config.TQF_USE_Z6_AUGMENTATION_DEFAULT, bool)
        self.assertTrue(config.TQF_USE_Z6_AUGMENTATION_DEFAULT)

    def test_orbit_mixing_temp_rotation_default(self) -> None:
        """Test rotation temperature default is 0.3 (sharp weighting)."""
        self.assertIsInstance(config.TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT, float)
        self.assertEqual(config.TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT, 0.3)

    def test_orbit_mixing_temp_reflection_default(self) -> None:
        """Test reflection temperature default is 0.5 (moderate weighting)."""
        self.assertIsInstance(config.TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT, float)
        self.assertEqual(config.TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT, 0.5)

    def test_orbit_mixing_temp_inversion_default(self) -> None:
        """Test inversion temperature default is 0.7 (soft weighting)."""
        self.assertIsInstance(config.TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT, float)
        self.assertEqual(config.TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT, 0.7)

    def test_orbit_mixing_temp_ordering(self) -> None:
        """Test that temperatures follow rotation < reflection < inversion.

        WHY: The mark 3 spec requires progressively softer weighting for
             more abstract symmetry operations. Rotation is most reliable
             (sharp), inversion is most abstract (soft).
        """
        self.assertLess(config.TQF_ORBIT_MIXING_TEMP_ROTATION_DEFAULT,
                        config.TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT)
        self.assertLess(config.TQF_ORBIT_MIXING_TEMP_REFLECTION_DEFAULT,
                        config.TQF_ORBIT_MIXING_TEMP_INVERSION_DEFAULT)

    def test_orbit_mixing_temp_range_constants(self) -> None:
        """Test orbit mixing temperature range constants have sensible values."""
        self.assertIsInstance(config.TQF_ORBIT_MIXING_TEMP_MIN, float)
        self.assertIsInstance(config.TQF_ORBIT_MIXING_TEMP_MAX, float)
        self.assertGreater(config.TQF_ORBIT_MIXING_TEMP_MIN, 0.0,
                          "Temperature MIN must be positive (avoid division by zero)")
        self.assertGreater(config.TQF_ORBIT_MIXING_TEMP_MAX, 1.0,
                          "Temperature MAX must allow values > 1.0 for uniform averaging")


class TestRemovedObsoleteConstants(unittest.TestCase):
    """Test that obsolete/unused constants have been removed."""

    def test_stochastic_depth_removed(self) -> None:
        """Test that unused STOCHASTIC_DEPTH_PROB_DEFAULT was removed."""
        self.assertFalse(hasattr(config, 'STOCHASTIC_DEPTH_PROB_DEFAULT'),
                        "STOCHASTIC_DEPTH_PROB_DEFAULT should have been removed (unused)")

    def test_cache_graph_structures_removed(self) -> None:
        """Test that unused TQF_CACHE_GRAPH_STRUCTURES_DEFAULT was removed."""
        self.assertFalse(hasattr(config, 'TQF_CACHE_GRAPH_STRUCTURES_DEFAULT'),
                        "TQF_CACHE_GRAPH_STRUCTURES_DEFAULT should have been removed (unused)")

    def test_max_fractal_gates_removed(self) -> None:
        """Test that unused TQF_MAX_FRACTAL_GATES_DEFAULT was removed."""
        self.assertFalse(hasattr(config, 'TQF_MAX_FRACTAL_GATES_DEFAULT'),
                        "TQF_MAX_FRACTAL_GATES_DEFAULT should have been removed (unused)")


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all configuration tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [
        TestReproducibilityConstants,
        TestDatasetSizeConfiguration,
        TestTQFArchitectureParameters,
        TestTrainingHyperparameters,
        TestHardwareConfiguration,
        TestParameterMatchingConfiguration,
        TestConfigurationConsistency,
        TestSNNParameters,
        TestRangeConstants,
        TestDefaultsWithinRanges,
        TestOrbitMixingAndAugmentationDefaults,
        TestRemovedObsoleteConstants
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Configuration Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE CONFIGURATION TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
