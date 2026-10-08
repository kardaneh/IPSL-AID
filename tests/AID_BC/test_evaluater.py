# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.evaluater import (
    MetricTracker,
    summary_metrics,
    evaluate,
)

# python -m unittest tests.test_evaluater


# ============================================================================
# Evaluater Test Suite
# ============================================================================


class TestMetricTracker(unittest.TestCase):
    """Unit tests for MetricTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up MetricTracker test fixtures")

    def test_init(self):
        """Test MetricTracker initialization."""
        if self.logger:
            self.logger.info("Testing MetricTracker initialization")

        tracker = MetricTracker()

        self.assertEqual(tracker.count, 0)
        self.assertEqual(tracker.sum, 0.0)
        self.assertEqual(tracker.sum_sq, 0.0)
        self.assertEqual(tracker.min, np.inf)
        self.assertEqual(tracker.max, -np.inf)

        if self.logger:
            self.logger.info("✅ MetricTracker initialization test passed")

    def test_update(self):
        """Test MetricTracker update."""
        if self.logger:
            self.logger.info("Testing MetricTracker update")

        tracker = MetricTracker()
        tracker.update(np.array([1.0, 2.0, 3.0]))

        self.assertEqual(tracker.count, 3)
        self.assertEqual(tracker.sum, 6.0)
        self.assertEqual(tracker.sum_sq, 14.0)
        self.assertEqual(tracker.min, 1.0)
        self.assertEqual(tracker.max, 3.0)

        if self.logger:
            self.logger.info("✅ MetricTracker update test passed")

    def test_update_ignores_non_finite_values(self):
        """Test that non-finite values are ignored."""
        if self.logger:
            self.logger.info("Testing non-finite value handling")

        tracker = MetricTracker()

        tracker.update(
            np.array(
                [
                    1.0,
                    np.nan,
                    np.inf,
                    -np.inf,
                    3.0,
                ]
            )
        )

        self.assertEqual(tracker.count, 2)
        self.assertEqual(tracker.sum, 4.0)
        self.assertEqual(tracker.min, 1.0)
        self.assertEqual(tracker.max, 3.0)

        if self.logger:
            self.logger.info("✅ Non-finite value handling test passed")

    def test_mean(self):
        """Test mean calculation."""
        if self.logger:
            self.logger.info("Testing MetricTracker mean")

        tracker = MetricTracker()
        tracker.update(np.array([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(
            tracker.getmean(),
            2.0,
        )

        if self.logger:
            self.logger.info("✅ MetricTracker mean test passed")

    def test_std(self):
        """Test standard deviation calculation."""
        if self.logger:
            self.logger.info("Testing MetricTracker standard deviation")

        values = np.array([1.0, 2.0, 3.0, 4.0])

        tracker = MetricTracker()
        tracker.update(values)

        self.assertAlmostEqual(
            tracker.getstd(),
            np.std(values),
        )

        if self.logger:
            self.logger.info("✅ MetricTracker standard deviation test passed")

    def test_min_max(self):
        """Test minimum and maximum calculations."""
        if self.logger:
            self.logger.info("Testing MetricTracker min/max")

        tracker = MetricTracker()
        tracker.update(np.array([-2.0, 5.0, 1.0]))

        self.assertEqual(
            tracker.getmin(),
            -2.0,
        )
        self.assertEqual(
            tracker.getmax(),
            5.0,
        )

        if self.logger:
            self.logger.info("✅ MetricTracker min/max test passed")

    def test_empty_tracker(self):
        """Test empty MetricTracker behavior."""
        if self.logger:
            self.logger.info("Testing empty MetricTracker")

        tracker = MetricTracker()

        self.assertTrue(np.isnan(tracker.getmean()))
        self.assertTrue(np.isnan(tracker.getstd()))
        self.assertTrue(np.isnan(tracker.getmin()))
        self.assertTrue(np.isnan(tracker.getmax()))

        if self.logger:
            self.logger.info("✅ Empty MetricTracker test passed")

    def test_sqrtmean(self):
        """Test square root of mean calculation."""
        if self.logger:
            self.logger.info("Testing MetricTracker square root of mean")

        tracker = MetricTracker()
        tracker.update(np.array([4.0, 16.0]))

        self.assertAlmostEqual(
            tracker.getsqrtmean(),
            np.sqrt(10.0),
        )

        if self.logger:
            self.logger.info("✅ MetricTracker square root of mean test passed")


class TestSummaryMetrics(unittest.TestCase):
    """Unit tests for summary_metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up summary_metrics test fixtures")

    def test_without_reference(self):
        """Test summary metrics without reference data."""
        if self.logger:
            self.logger.info("Testing summary_metrics without reference")

        data = np.array([1.0, 2.0, 3.0])

        result = summary_metrics(
            data=data,
            name="ERA5",
            variable_name="VAR_2T",
        )

        self.assertEqual(
            result["dataset"],
            "ERA5",
        )
        self.assertAlmostEqual(
            result["MEAN"],
            2.0,
        )
        self.assertAlmostEqual(
            result["STD"],
            np.std(data),
        )
        self.assertEqual(
            result["MIN"],
            1.0,
        )
        self.assertEqual(
            result["MAX"],
            3.0,
        )
        self.assertTrue(np.isnan(result["MAE"]))
        self.assertTrue(np.isnan(result["RMSE"]))

        if self.logger:
            self.logger.info("✅ summary_metrics without reference test passed")

    def test_with_reference(self):
        """Test summary metrics with reference data."""
        if self.logger:
            self.logger.info("Testing summary_metrics with reference")

        reference = np.array([1.0, 2.0, 3.0])
        data = np.array([2.0, 2.0, 5.0])

        result = summary_metrics(
            data=data,
            name="CMIP6_raw",
            variable_name="VAR_2T",
            ref=reference,
        )

        error = data - reference

        expected_mae = np.mean(np.abs(error))
        expected_rmse = np.sqrt(np.mean(error**2))

        self.assertAlmostEqual(
            result["MAE"],
            expected_mae,
        )
        self.assertAlmostEqual(
            result["RMSE"],
            expected_rmse,
        )

        if self.logger:
            self.logger.info("✅ summary_metrics with reference test passed")

    def test_precipitation_conversion(self):
        """Test precipitation conversion from m to mm."""
        if self.logger:
            self.logger.info("Testing precipitation conversion")

        reference = np.array([0.001, 0.002, 0.003])
        data = np.array([0.002, 0.002, 0.005])

        result = summary_metrics(
            data=data,
            name="CMIP6_raw",
            variable_name="tp",
            ref=reference,
        )

        data_mm = data * 1000.0
        reference_mm = reference * 1000.0
        error_mm = data_mm - reference_mm

        self.assertAlmostEqual(
            result["MEAN"],
            np.mean(data_mm),
        )
        self.assertAlmostEqual(
            result["STD"],
            np.std(data_mm),
        )
        self.assertAlmostEqual(
            result["MIN"],
            np.min(data_mm),
        )
        self.assertAlmostEqual(
            result["MAX"],
            np.max(data_mm),
        )
        self.assertAlmostEqual(
            result["MAE"],
            np.mean(np.abs(error_mm)),
        )
        self.assertAlmostEqual(
            result["RMSE"],
            np.sqrt(np.mean(error_mm**2)),
        )

        if self.logger:
            self.logger.info("✅ Precipitation conversion test passed")


class TestEvaluate(unittest.TestCase):
    """Integration test for evaluate."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.variable_names = [
            "VAR_2T",
            "VAR_10U",
        ]

        time = pd.date_range(
            "2021-01-01",
            periods=4,
            freq="h",
        )

        latitude = np.linspace(
            40.0,
            45.0,
            4,
        )

        longitude = np.linspace(
            -5.0,
            5.0,
            5,
        )

        shape = (
            len(time),
            len(latitude),
            len(longitude),
        )

        coords = {
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
        }

        reference_data = {}
        raw_data = {}
        corrected_data = {}

        rng = np.random.default_rng(42)

        for variable_name in self.variable_names:
            reference_values = rng.normal(size=shape)
            raw_values = reference_values + 1.0
            corrected_values = reference_values + 0.2

            reference_data[variable_name] = (
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
                reference_values,
            )

            raw_data[variable_name] = (
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
                raw_values,
            )

            corrected_data[variable_name] = (
                (
                    "time",
                    "latitude",
                    "longitude",
                ),
                corrected_values,
            )

        self.reference = xr.Dataset(
            reference_data,
            coords=coords,
        )

        self.raw = xr.Dataset(
            raw_data,
            coords=coords,
        )

        self.corrected = xr.Dataset(
            corrected_data,
            coords=coords,
        )

        if self.logger:
            self.logger.info(
                "Test setup complete - "
                f"Variables: {len(self.variable_names)}, "
                f"Time steps: {len(time)}, "
                f"Resolution: {len(latitude)}x{len(longitude)}"
            )

    @patch("AID_BC.evaluater.plot_surface")
    @patch("AID_BC.evaluater.plot_qq_quantiles")
    @patch("AID_BC.evaluater.plot_power_spectra")
    @patch("AID_BC.evaluater.plot_validation_pdfs")
    @patch("AID_BC.evaluater.plot_metrics_heatmap")
    def test_evaluate(
        self,
        mock_heatmap,
        mock_pdf,
        mock_spectra,
        mock_qq,
        mock_surface,
    ):
        """Test complete evaluation workflow."""
        if self.logger:
            self.logger.info("Testing complete evaluation workflow")

        mock_surface.return_value = "surface.png"

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluate(
                reference=self.reference,
                raw=self.raw,
                corrected=self.corrected,
                variable_names=self.variable_names,
                method="ot",
                year=2021,
                logger=self.logger,
                run_name="test_run",
                results_dir=tmpdir,
            )

            result_dir = os.path.join(
                tmpdir,
                "test_run",
            )

            self.assertTrue(os.path.isdir(result_dir))

            # Two heatmaps per variable
            self.assertEqual(
                mock_heatmap.call_count,
                2 * len(self.variable_names),
            )

            self.assertEqual(
                mock_pdf.call_count,
                1,
            )

            self.assertEqual(
                mock_spectra.call_count,
                1,
            )

            self.assertEqual(
                mock_qq.call_count,
                1,
            )

            # min(3, 4 time steps) = 3
            self.assertEqual(
                mock_surface.call_count,
                3,
            )

            # Each surface plot must contain exactly one time step
            for call in mock_surface.call_args_list:
                predictions = call.kwargs["predictions"]
                targets = call.kwargs["targets"]
                coarse_inputs = call.kwargs["coarse_inputs"]

                for array in predictions:
                    self.assertEqual(
                        array.sizes["time"],
                        1,
                    )

                for array in targets:
                    self.assertEqual(
                        array.sizes["time"],
                        1,
                    )

                for array in coarse_inputs:
                    self.assertEqual(
                        array.sizes["time"],
                        1,
                    )

        if self.logger:
            self.logger.info("✅ Complete evaluation workflow test passed")


def run_tests():
    """Run all evaluater tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMetricTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestSummaryMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluate))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
