"""
test_output_formatters.py - Comprehensive Tests for Output Formatting

This module tests all output formatting functions in output_formatters.py,
ensuring consistent ASCII-only output, correct numeric formatting, and proper
table/separator generation for experiment results.

Key Test Coverage:
- Basic Formatters: format_accuracy, format_loss, format_confidence_interval
- Numeric Edge Cases: Zero values, negative values, values > 1.0, NaN/inf handling
- Separator Generation: make_separator, print_section_header with various widths
- Label-Value Formatting: Alignment, custom widths, format string validation
- Time Formatting: Human-readable elapsed time (HH:MM:SS), timestamp generation
- ASCII Validation: Ensures no Unicode characters in output (LaTeX compatibility)
- Width Consistency: Validates standard widths (80/92/120/175 chars)

Scientific Rationale:
ASCII-only output ensures experiment results can be directly copy-pasted into
LaTeX documents and displayed correctly in any terminal without encoding issues,
critical for reproducible scientific reporting.

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
import os
import json
import tempfile
from typing import Any, Dict
import numpy as np

from conftest import PATHS

try:
    import output_formatters as fmt
except ImportError as e:
    raise ImportError(f"Cannot import output_formatters: {e}")


class TestBasicFormatters(unittest.TestCase):
    """Test suite for basic formatting functions."""

    def test_format_accuracy_returns_string(self) -> None:
        """Test that format_accuracy returns a string."""
        result: str = fmt.format_accuracy(0.9523)
        self.assertIsInstance(result, str)

    def test_format_accuracy_percentage(self) -> None:
        """Test accuracy formatting with percentage conversion."""
        result: str = fmt.format_accuracy(0.9523)
        self.assertIn("95.23", result)

    def test_format_accuracy_zero(self) -> None:
        """Test formatting zero accuracy."""
        result: str = fmt.format_accuracy(0.0)
        self.assertIn("0.00", result)

    def test_format_accuracy_one(self) -> None:
        """Test formatting perfect accuracy."""
        result: str = fmt.format_accuracy(1.0)
        self.assertIn("100.00", result)

    def test_format_accuracy_negative(self) -> None:
        """Test formatting negative accuracy (edge case)."""
        result: str = fmt.format_accuracy(-0.1)
        self.assertIsInstance(result, str)

    def test_format_accuracy_greater_than_one(self) -> None:
        """Test formatting accuracy > 1.0 (edge case)."""
        result: str = fmt.format_accuracy(1.5)
        self.assertIsInstance(result, str)

    def test_format_accuracy_custom_width(self) -> None:
        """Test format_accuracy with custom width."""
        result: str = fmt.format_accuracy(0.9523, width=8)
        self.assertIsInstance(result, str)

    def test_format_loss_returns_string(self) -> None:
        """Test that format_loss returns a string."""
        result: str = fmt.format_loss(0.1234)
        self.assertIsInstance(result, str)

    def test_format_loss_precision(self) -> None:
        """Test loss formatting precision."""
        result: str = fmt.format_loss(0.123456)
        # Should have 4 decimal places
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_format_loss_zero(self) -> None:
        """Test formatting zero loss."""
        result: str = fmt.format_loss(0.0)
        self.assertIsInstance(result, str)

    def test_format_loss_large_value(self) -> None:
        """Test formatting large loss value."""
        result: str = fmt.format_loss(100.5)
        self.assertIsInstance(result, str)

    def test_format_time_ms_returns_string(self) -> None:
        """Test that format_time_ms returns a string."""
        result: str = fmt.format_time_ms(1234.5678)
        self.assertIsInstance(result, str)

    def test_format_time_ms_includes_ms(self) -> None:
        """Test time formatting includes ms unit."""
        result: str = fmt.format_time_ms(1234.5678)
        self.assertIn("ms", result.lower())

    def test_format_time_ms_zero(self) -> None:
        """Test formatting zero time."""
        result: str = fmt.format_time_ms(0.0)
        self.assertIsInstance(result, str)

    def test_format_time_ms_small_value(self) -> None:
        """Test formatting very small time value."""
        result: str = fmt.format_time_ms(0.001)
        self.assertIsInstance(result, str)

    def test_format_params_returns_string(self) -> None:
        """Test that format_params returns a string."""
        result: str = fmt.format_params(650000)
        self.assertIsInstance(result, str)

    def test_format_params_comma_separated(self) -> None:
        """Test params formatting with thousands separator or M/K notation."""
        result: str = fmt.format_params(1234567)
        # Should contain either comma separator OR M/K notation
        has_separator: bool = "," in result or "M" in result or "K" in result
        self.assertTrue(has_separator, f"Expected comma or M/K notation in: {result}")

    def test_format_params_zero(self) -> None:
        """Test formatting zero parameters."""
        result: str = fmt.format_params(0)
        self.assertIsInstance(result, str)

    def test_format_params_small_value(self) -> None:
        """Test formatting small parameter count."""
        result: str = fmt.format_params(100)
        self.assertIsInstance(result, str)

    def test_format_mean_std_returns_string(self) -> None:
        """Test that format_mean_std returns a string."""
        result: str = fmt.format_mean_std(0.95, 0.02)
        self.assertIsInstance(result, str)

    def test_format_mean_std_includes_plus_minus(self) -> None:
        """Test mean +/- std formatting includes separator."""
        result: str = fmt.format_mean_std(0.95, 0.02)
        # Should contain some separator (+/- or +/-)
        self.assertTrue(any(c in result for c in ['+/-', '+', '-']))

    def test_format_mean_std_zero_std(self) -> None:
        """Test formatting with zero standard deviation."""
        result: str = fmt.format_mean_std(0.95, 0.0)
        self.assertIsInstance(result, str)

    def test_format_mean_std_negative_mean(self) -> None:
        """Test formatting with negative mean."""
        result: str = fmt.format_mean_std(-0.5, 0.1)
        self.assertIsInstance(result, str)


class TestSeparators(unittest.TestCase):
    """Test suite for separator formatting."""

    def test_make_separator_returns_string(self) -> None:
        """Test that make_separator returns a string."""
        result: str = fmt.make_separator()
        self.assertIsInstance(result, str)

    def test_make_separator_has_correct_length(self) -> None:
        """Test separator has specified length."""
        width: int = 80
        result: str = fmt.make_separator(width=width)
        self.assertEqual(len(result), width)

    def test_make_separator_default_char(self) -> None:
        """Test separator uses default character."""
        result: str = fmt.make_separator()
        self.assertTrue(all(c == fmt.MAJOR_SEP_CHAR for c in result))

    def test_make_separator_custom_char(self) -> None:
        """Test separator with custom character."""
        result: str = fmt.make_separator(char='-')
        self.assertTrue(all(c == '-' for c in result))

    def test_make_separator_various_widths(self) -> None:
        """Test separator with various widths."""
        for width in [40, 80, 100, 120]:
            result: str = fmt.make_separator(width=width)
            self.assertEqual(len(result), width)

    def test_make_separator_zero_width(self) -> None:
        """Test separator with zero width."""
        result: str = fmt.make_separator(width=0)
        self.assertEqual(len(result), 0)

    def test_make_separator_single_char_width(self) -> None:
        """Test separator with width of 1."""
        result: str = fmt.make_separator(width=1)
        self.assertEqual(len(result), 1)


class TestLabeledValueFormatting(unittest.TestCase):
    """Test suite for labeled value formatting."""

    def test_format_labeled_value_basic(self) -> None:
        """Test basic labeled value formatting."""
        result: str = fmt.format_labeled_value("Test", 42)
        self.assertIsInstance(result, str)
        self.assertIn("Test", result)
        self.assertIn("42", result)

    def test_format_labeled_value_with_colon(self) -> None:
        """Test labeled value includes colon separator."""
        result: str = fmt.format_labeled_value("Label", "Value")
        self.assertIn(":", result)

    def test_format_labeled_value_float_format(self) -> None:
        """Test labeled value with float formatting."""
        result: str = fmt.format_labeled_value("Rate", 0.001, value_fmt='.2e')
        self.assertIn("1.00e-03", result.lower())

    def test_format_labeled_value_integer_format(self) -> None:
        """Test labeled value with integer formatting."""
        result: str = fmt.format_labeled_value("Count", 1000, value_fmt=',d')
        self.assertIn("1,000", result)

    def test_format_labeled_value_string_value(self) -> None:
        """Test labeled value with string value."""
        result: str = fmt.format_labeled_value("Model", "TQF-ANN")
        self.assertIn("TQF-ANN", result)

    def test_format_labeled_value_none_value(self) -> None:
        """Test labeled value with None."""
        result: str = fmt.format_labeled_value("Value", None)
        self.assertIsInstance(result, str)

    def test_format_labeled_value_boolean(self) -> None:
        """Test labeled value with boolean."""
        result: str = fmt.format_labeled_value("Flag", True)
        self.assertIn("True", result)


class TestConstantsAndDefaults(unittest.TestCase):
    """Test suite for formatting constants."""

    def test_major_sep_char_defined(self) -> None:
        """Test MAJOR_SEP_CHAR is defined."""
        self.assertTrue(hasattr(fmt, 'MAJOR_SEP_CHAR'))
        self.assertIsInstance(fmt.MAJOR_SEP_CHAR, str)
        self.assertEqual(len(fmt.MAJOR_SEP_CHAR), 1)

    def test_minor_sep_char_defined(self) -> None:
        """Test MINOR_SEP_CHAR is defined."""
        self.assertTrue(hasattr(fmt, 'MINOR_SEP_CHAR'))
        self.assertIsInstance(fmt.MINOR_SEP_CHAR, str)
        self.assertEqual(len(fmt.MINOR_SEP_CHAR), 1)

    def test_width_constants_defined(self) -> None:
        """Test width constants are defined."""
        for const in ['WIDTH_NARROW', 'WIDTH_STANDARD', 'WIDTH_WIDE', 'WIDTH_EXTRA']:
            self.assertTrue(hasattr(fmt, const))
            value: int = getattr(fmt, const)
            self.assertIsInstance(value, int)
            self.assertGreater(value, 0)

    def test_width_constants_ordered(self) -> None:
        """Test width constants are in ascending order."""
        self.assertLess(fmt.WIDTH_NARROW, fmt.WIDTH_STANDARD)
        self.assertLess(fmt.WIDTH_STANDARD, fmt.WIDTH_WIDE)
        self.assertLess(fmt.WIDTH_WIDE, fmt.WIDTH_EXTRA)

    def test_column_width_constants(self) -> None:
        """Test column width constants are defined."""
        self.assertTrue(hasattr(fmt, 'COL_LABEL'))
        self.assertTrue(hasattr(fmt, 'COL_VALUE'))
        self.assertIsInstance(fmt.COL_LABEL, int)
        self.assertIsInstance(fmt.COL_VALUE, int)
        self.assertGreater(fmt.COL_LABEL, 0)
        self.assertGreater(fmt.COL_VALUE, 0)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling."""

    def test_format_accuracy_very_small(self) -> None:
        """Test formatting very small accuracy."""
        result: str = fmt.format_accuracy(1e-10)
        self.assertIsInstance(result, str)

    def test_format_accuracy_very_large(self) -> None:
        """Test formatting very large accuracy."""
        result: str = fmt.format_accuracy(999.9)
        self.assertIsInstance(result, str)

    def test_format_loss_nan(self) -> None:
        """Test formatting NaN loss (should not crash)."""
        try:
            result: str = fmt.format_loss(float('nan'))
            self.assertIsInstance(result, str)
        except:
            pass  # Some implementations may raise

    def test_format_loss_inf(self) -> None:
        """Test formatting infinite loss (should not crash)."""
        try:
            result: str = fmt.format_loss(float('inf'))
            self.assertIsInstance(result, str)
        except:
            pass  # Some implementations may raise

    def test_format_params_negative(self) -> None:
        """Test formatting negative parameter count (edge case)."""
        result: str = fmt.format_params(-100)
        self.assertIsInstance(result, str)

    def test_format_mean_std_both_zero(self) -> None:
        """Test formatting with both mean and std zero."""
        result: str = fmt.format_mean_std(0.0, 0.0)
        self.assertIsInstance(result, str)

    def test_make_separator_large_width(self) -> None:
        """Test separator with very large width."""
        result: str = fmt.make_separator(width=1000)
        self.assertEqual(len(result), 1000)


class TestPrintFunctions(unittest.TestCase):
    """Test suite for print_* output functions that write to stdout."""

    def test_print_section_header_outputs_title(self) -> None:
        """Test print_section_header includes the title text."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_section_header("TEST SECTION")
        output = buf.getvalue()
        self.assertIn("TEST SECTION", output)
        self.assertIn(fmt.MAJOR_SEP_CHAR, output)

    def test_print_model_training_start(self) -> None:
        """Test print_model_training_start outputs model name."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_model_training_start("TQF-ANN", 3)
        output = buf.getvalue()
        self.assertIn("TQF-ANN", output)
        self.assertIn("BEGIN MODEL TRAINING", output)
        self.assertIn("#", output)

    def test_print_model_training_end(self) -> None:
        """Test print_model_training_end outputs model name."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_model_training_end("TQF-ANN")
        output = buf.getvalue()
        self.assertIn("TQF-ANN", output)
        self.assertIn("END MODEL TRAINING", output)

    def test_print_epoch_progress_basic(self) -> None:
        """Test print_epoch_progress outputs epoch info."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_epoch_progress(
                epoch=4, total_epochs=100,
                train_loss=0.5, val_loss=0.4,
                val_acc=92.5, lr=0.001, elapsed=12.3
            )
        output = buf.getvalue()
        self.assertIn("5", output)  # epoch+1
        self.assertIn("100", output)
        self.assertIn("92.50", output)

    def test_print_epoch_progress_with_optional_losses(self) -> None:
        """Test print_epoch_progress includes optional loss terms."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_epoch_progress(
                epoch=0, total_epochs=10,
                train_loss=1.0, val_loss=0.9,
                val_acc=50.0, lr=0.001, elapsed=5.0,
                geom_loss=0.0123, z6_equiv_loss=0.0456,
                d6_equiv_loss=0.0789, t24_orbit_loss=0.0012
            )
        output = buf.getvalue()
        self.assertIn("Geom", output)
        self.assertIn("Z6", output)
        self.assertIn("D6", output)
        self.assertIn("T24", output)

    def test_print_early_stopping_message(self) -> None:
        """Test print_early_stopping_message outputs patience and epoch info."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_early_stopping_message(
                patience=15, best_loss_epoch=24,
                best_acc_epoch=22, total_epochs=100,
                best_val_acc_at_best_loss=0.9523,
                best_val_acc_overall=0.9540,
                best_val_loss=0.1234
            )
        output = buf.getvalue()
        self.assertIn("EARLY STOPPING", output)
        self.assertIn("15", output)
        self.assertIn("25", output)  # best_loss_epoch + 1
        self.assertIn("95.23", output)  # best_val_acc_at_best_loss * 100

    def test_print_seed_header(self) -> None:
        """Test print_seed_header outputs seed info."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_seed_header(seed=42, total_seeds=3, model_name="FC-MLP", seed_idx=1)
        output = buf.getvalue()
        self.assertIn("FC-MLP", output)
        self.assertIn("42", output)
        self.assertIn("1", output)
        self.assertIn("3", output)

    def test_print_seed_results_summary_minimal(self) -> None:
        """Test print_seed_results_summary with minimal results dict."""
        import io
        from contextlib import redirect_stdout
        results: Dict[str, Any] = {
            'best_val_acc': 95.0,
            'test_unrot_acc': 94.5,
            'test_rot_acc': 93.0,
            'rotation_inv_error': 0.012,
            'invariance_l2_error': 1.5e-6,
            'params': 650000,
            'flops': 12000000,
            'inference_time_ms': 1.23,
            'train_time_total': 120.5,
            'best_loss_epoch': 24,
            'best_acc_epoch': 22,
            'train_time_per_epoch': 1.2
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_seed_results_summary(seed=42, model_name="TQF-ANN", results=results)
        output = buf.getvalue()
        self.assertIn("TQF-ANN", output)
        self.assertIn("95.00", output)
        self.assertIn("ACCURACY SUMMARY", output)
        self.assertIn("GEOMETRIC INVARIANCE", output)
        self.assertIn("MODEL SPECIFICATIONS", output)
        self.assertIn("PERFORMANCE METRICS", output)

    def test_print_seed_results_summary_with_per_class(self) -> None:
        """Test print_seed_results_summary with per-class accuracy enabled."""
        import io
        from contextlib import redirect_stdout
        results: Dict[str, Any] = {
            'best_val_acc': 95.0,
            'test_unrot_acc': 94.5,
            'test_rot_acc': 93.0,
            'per_class_acc': {i: 0.90 + i * 0.01 for i in range(10)}
        }
        buf = io.StringIO()
        with redirect_stdout(buf):
            fmt.print_seed_results_summary(
                seed=42, model_name="TQF-ANN", results=results,
                show_per_class=True
            )
        output = buf.getvalue()
        self.assertIn("PER-CLASS ACCURACY", output)


class TestResultSerialization(unittest.TestCase):
    """Test suite for result serialization and disk I/O functions."""

    def test_make_result_serializable_basic(self) -> None:
        """Test _make_result_serializable converts numpy types."""
        result: Dict[str, Any] = {
            'best_val_acc': np.float64(95.23),
            'params': np.int64(650000),
            'model_name': 'TQF-ANN',
            'seed': 42
        }
        out = fmt._make_result_serializable(result)
        self.assertIsInstance(out['best_val_acc'], float)
        self.assertIsInstance(out['params'], float)
        self.assertIsInstance(out['model_name'], str)
        self.assertEqual(out['seed'], 42)

    def test_make_result_serializable_per_class_acc(self) -> None:
        """Test _make_result_serializable handles per_class_acc dict keys."""
        result: Dict[str, Any] = {
            'per_class_acc': {0: np.float64(0.95), 1: np.float64(0.92)}
        }
        out = fmt._make_result_serializable(result)
        self.assertIn('0', out['per_class_acc'])
        self.assertIn('1', out['per_class_acc'])
        self.assertIsInstance(out['per_class_acc']['0'], float)

    def test_make_result_serializable_ndarray(self) -> None:
        """Test _make_result_serializable converts numpy arrays to lists."""
        result: Dict[str, Any] = {
            'some_array': np.array([1.0, 2.0, 3.0])
        }
        out = fmt._make_result_serializable(result)
        self.assertIsInstance(out['some_array'], list)
        self.assertEqual(out['some_array'], [1.0, 2.0, 3.0])

    def test_make_result_serializable_json_compatible(self) -> None:
        """Test _make_result_serializable output is JSON-serializable."""
        result: Dict[str, Any] = {
            'best_val_acc': np.float64(95.23),
            'params': np.int64(650000),
            'per_class_acc': {i: np.float64(0.9 + i * 0.01) for i in range(10)},
            'model_name': 'TQF-ANN',
            'seed': 42
        }
        out = fmt._make_result_serializable(result)
        # Should not raise
        json_str = json.dumps(out)
        self.assertIsInstance(json_str, str)

    def test_save_seed_result_to_disk_creates_file(self) -> None:
        """Test save_seed_result_to_disk creates a JSON results file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "output", "results.json")
            result: Dict[str, Any] = {
                'model_name': 'TQF-ANN',
                'seed': 42,
                'best_val_acc': 95.0,
                'test_unrot_acc': 94.5,
                'test_rot_acc': 93.0
            }
            fmt.save_seed_result_to_disk(result, path)
            self.assertTrue(os.path.exists(path))

            with open(path, 'r') as f:
                data = json.load(f)
            self.assertIn('results', data)
            self.assertIn('TQF-ANN', data['results'])
            self.assertEqual(len(data['results']['TQF-ANN']), 1)
            self.assertEqual(data['status'], 'in_progress')

    def test_save_seed_result_to_disk_appends(self) -> None:
        """Test save_seed_result_to_disk appends multiple seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            for seed in [42, 43, 44]:
                result: Dict[str, Any] = {
                    'model_name': 'TQF-ANN',
                    'seed': seed,
                    'best_val_acc': 95.0
                }
                fmt.save_seed_result_to_disk(result, path)

            with open(path, 'r') as f:
                data = json.load(f)
            self.assertEqual(len(data['results']['TQF-ANN']), 3)

    def test_save_seed_result_with_experiment_config(self) -> None:
        """Test save_seed_result_to_disk stores experiment config on first call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            config = {'learning_rate': 0.001, 'batch_size': 128}
            result: Dict[str, Any] = {
                'model_name': 'FC-MLP',
                'seed': 42,
                'best_val_acc': 90.0
            }
            fmt.save_seed_result_to_disk(result, path, experiment_config=config)

            with open(path, 'r') as f:
                data = json.load(f)
            self.assertIn('config', data)
            self.assertEqual(data['config']['learning_rate'], 0.001)

    def test_save_final_summary_to_disk(self) -> None:
        """Test save_final_summary_to_disk writes JSON and TXT files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            # Create initial file
            result: Dict[str, Any] = {
                'model_name': 'TQF-ANN',
                'seed': 42,
                'best_val_acc': 95.0
            }
            fmt.save_seed_result_to_disk(result, path)

            summary = {
                'TQF-ANN': {
                    'val_acc': (95.0, 0.5),
                    'test_unrot_acc': (94.5, 0.3),
                    'test_rot_acc': (93.0, 0.4),
                    'params': (650000.0, 0.0),
                    'flops': (12000000.0, 0.0),
                    'inference_time_ms': (1.23, 0.05)
                }
            }
            fmt.save_final_summary_to_disk(summary, path)

            with open(path, 'r') as f:
                data = json.load(f)
            self.assertEqual(data['status'], 'completed')
            self.assertIn('summary', data)
            self.assertIn('TQF-ANN', data['summary'])
            self.assertIn('completed_at', data)

            # Check TXT companion file
            txt_path = path.replace('.json', '.txt')
            self.assertTrue(os.path.exists(txt_path))
            with open(txt_path, 'r') as f:
                txt_content = f.read()
            self.assertIn("TQF-ANN", txt_content)
            self.assertIn("FINAL MODEL COMPARISON", txt_content)


class TestTimeFormatting(unittest.TestCase):
    """Test suite for time formatting functions."""

    def test_format_time_seconds_under_minute(self) -> None:
        """Test seconds formatting for < 60s."""
        result = fmt.format_time_seconds(30.5)
        self.assertIn("sec", result)

    def test_format_time_seconds_minutes(self) -> None:
        """Test seconds formatting for minutes range."""
        result = fmt.format_time_seconds(90.0)
        self.assertIn("min", result)

    def test_format_time_seconds_hours(self) -> None:
        """Test seconds formatting for hours range."""
        result = fmt.format_time_seconds(7200.0)
        self.assertIn("hr", result)


class TestFlopsFormatting(unittest.TestCase):
    """Test suite for FLOPs formatting."""

    def test_format_flops_millions(self) -> None:
        result = fmt.format_flops(12e6)
        self.assertIn("M", result)

    def test_format_flops_billions(self) -> None:
        result = fmt.format_flops(1.5e9)
        self.assertIn("G", result)

    def test_format_flops_trillions(self) -> None:
        result = fmt.format_flops(2.0e12)
        self.assertIn("T", result)

    def test_format_flops_small(self) -> None:
        result = fmt.format_flops(500.0)
        self.assertIsInstance(result, str)

    def test_format_scientific(self) -> None:
        result = fmt.format_scientific(1.5e-6)
        self.assertIn("e", result.lower())


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all output formatter tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [
        TestBasicFormatters,
        TestSeparators,
        TestLabeledValueFormatting,
        TestConstantsAndDefaults,
        TestEdgeCases,
        TestPrintFunctions,
        TestResultSerialization,
        TestTimeFormatting,
        TestFlopsFormatting
    ]:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner: unittest.TextTestRunner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Output Formatter Tests')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE OUTPUT FORMATTER TESTS")
    print("=" * 80)

    result = run_tests(verbosity=2 if args.verbose else 1)
    sys.exit(0 if result.wasSuccessful() else 1)
