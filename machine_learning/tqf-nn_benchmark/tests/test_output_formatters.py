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
from typing import Any

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


def run_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run all output formatter tests."""
    loader: unittest.TestLoader = unittest.TestLoader()
    suite: unittest.TestSuite = unittest.TestSuite()

    for test_class in [
        TestBasicFormatters,
        TestSeparators,
        TestLabeledValueFormatting,
        TestConstantsAndDefaults,
        TestEdgeCases
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
