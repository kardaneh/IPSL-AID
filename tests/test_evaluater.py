# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

# ruff: noqa: E731
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from IPSL_AID.logger import Logger
from IPSL_AID.evaluater import (
    MetricTracker,
    mae_all,
    nmae_all,
    rmse_all,
    r2_all,
    pearson_all,
    kl_divergence_all,
    crps_ensemble_all,
    denormalize,
    run_validation,
    generate_residuals_norm,
)

import unittest
from unittest.mock import Mock, patch

# ============================================================================
# Evaluater Test Suite
# ============================================================================


class TestMetricTracker(unittest.TestCase):
    """Unit tests for MetricTracker class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up MetricTracker test fixtures")

    def test_metric_tracker_init(self):
        """Test MetricTracker initialization."""
        if self.logger:
            self.logger.info("Testing MetricTracker initialization")

        tracker = MetricTracker()

        self.assertEqual(tracker.value, 0.0)
        self.assertEqual(tracker.count, 0)

        if self.logger:
            self.logger.info("✅ MetricTracker initialization test passed")

    def test_metric_tracker_reset(self):
        """Test MetricTracker reset method."""
        if self.logger:
            self.logger.info("Testing MetricTracker reset")

        tracker = MetricTracker()
        tracker.value = 10.5
        tracker.count = 5

        tracker.reset()

        self.assertEqual(tracker.value, 0.0)
        self.assertEqual(tracker.count, 0)

        if self.logger:
            self.logger.info("✅ MetricTracker reset test passed")

    def test_metric_tracker_update(self):
        """Test MetricTracker update method."""
        if self.logger:
            self.logger.info("Testing MetricTracker update")

        tracker = MetricTracker()

        # First update
        tracker.update(10.0, 5)
        self.assertEqual(tracker.value, 50.0)  # 10 * 5
        self.assertEqual(tracker.count, 5)

        # Second update
        tracker.update(20.0, 3)
        self.assertEqual(tracker.value, 110.0)  # 50 + 20*3
        self.assertEqual(tracker.count, 8)  # 5 + 3

        # Third update with zero count
        tracker.update(30.0, 0)
        self.assertEqual(tracker.value, 110.0)  # Unchanged
        self.assertEqual(tracker.count, 8)  # Unchanged

        if self.logger:
            self.logger.info("✅ MetricTracker update test passed")

    def test_metric_tracker_getmean(self):
        """Test MetricTracker getmean method."""
        if self.logger:
            self.logger.info("Testing MetricTracker getmean")

        tracker = MetricTracker()

        # Test with valid updates
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        mean = tracker.getmean()
        expected_mean = 110.0 / 8  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
        self.assertAlmostEqual(mean, expected_mean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getmean()

        if self.logger:
            self.logger.info("✅ MetricTracker getmean test passed")

    def test_metric_tracker_getstd(self):
        """Test MetricTracker getstd method."""
        if self.logger:
            self.logger.info("Testing MetricTracker getstd")

        tracker = MetricTracker()

        # Known values
        # Values: [10 (×5), 20 (×3)]
        # mean = 13.75
        # E[x^2] = (10^2 * 5 + 20^2 * 3) / 8 = (500 + 1200) / 8 = 212.5
        # variance = 212.5 - 13.75^2 = 23.4375
        # std = sqrt(23.4375) ≈ 4.841229
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        std = tracker.getstd()
        expected_std = np.sqrt(212.5 - 13.75**2)

        self.assertAlmostEqual(std, expected_std, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getstd()

        if self.logger:
            self.logger.info("✅ MetricTracker getstd test passed")

    def test_metric_tracker_getsqrtmean(self):
        """Test MetricTracker getsqrtmean method."""
        if self.logger:
            self.logger.info("Testing MetricTracker getsqrtmean")

        tracker = MetricTracker()

        tracker.update(16.0, 2)  # mean = 16, sqrt = 4
        tracker.update(4.0, 2)  # mean = (16*2 + 4*2)/4 = 10, sqrt = sqrt(10)

        sqrtmean = tracker.getsqrtmean()
        expected_sqrtmean = np.sqrt(10.0)  # sqrt(10) ≈ 3.16227766
        self.assertAlmostEqual(sqrtmean, expected_sqrtmean, places=6)

        # Test with zero count (should raise ZeroDivisionError)
        tracker.reset()
        with self.assertRaises(ZeroDivisionError):
            tracker.getsqrtmean()

        if self.logger:
            self.logger.info("✅ MetricTracker getsqrtmean test passed")

    def test_metric_tracker_example_from_docstring(self):
        """Test the example provided in the docstring."""
        if self.logger:
            self.logger.info("Testing MetricTracker docstring example")

        tracker = MetricTracker()
        tracker.update(10.0, 5)
        tracker.update(20.0, 3)

        mean = tracker.getmean()
        sqrtmean = tracker.getsqrtmean()

        # Expected values from docstring example
        expected_mean = 110.0 / 8  # 13.75
        expected_sqrtmean = np.sqrt(13.75)  # 3.7080992435478315

        self.assertAlmostEqual(mean, expected_mean, places=6)
        self.assertAlmostEqual(sqrtmean, expected_sqrtmean, places=6)

        if self.logger:
            self.logger.info("✅ MetricTracker docstring example test passed")


class TestErrorMetrics(unittest.TestCase):
    """Unit tests for error metrics."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up error metrics test fixtures")

        self.metrics = {
            "MAE": mae_all,
            "NMAE": nmae_all,
            "RMSE": rmse_all,
            "R2": r2_all,
            "PEARSON": pearson_all,
        }

    def _compute_expected(self, metric_name, pred, true):
        mae = torch.mean(torch.abs(pred - true))
        if metric_name == "MAE":
            return mae

        elif metric_name == "NMAE":
            denom = torch.mean(torch.abs(true))
            return mae / denom if denom != 0 else torch.zeros_like(mae)

        elif metric_name == "RMSE":
            diff = pred - true
            return torch.sqrt(torch.mean(diff**2))

        elif metric_name == "R2":
            true_flat = true.reshape(-1)
            pred_flat = pred.reshape(-1)

            ss_res = torch.sum((true_flat - pred_flat) ** 2)
            ss_tot = torch.sum((true_flat - torch.mean(true_flat)) ** 2)
            return 1.0 - ss_res / (ss_tot + 1e-12)

        elif metric_name == "PEARSON":
            pred_flat = pred.reshape(-1)
            true_flat = true.reshape(-1)

            stacked = torch.stack([pred_flat, true_flat], dim=0)
            corr = torch.corrcoef(stacked)[0, 1]
            return corr

        else:
            raise ValueError(metric_name)

    def test_basic(self):
        """Test error metrics with simple tensors."""
        if self.logger:
            self.logger.info("Testing error metrics basic functionality")

        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.1, 1.9, 3.2])

        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                num_elements, value = func(pred, true)
                expected = self._compute_expected(name, pred, true)

                self.assertEqual(num_elements, 3)
                self.assertAlmostEqual(value.item(), expected.item(), places=4)

        if self.logger:
            self.logger.info("✅ Error metrics basic test passed")

    def test_exact_match(self):
        """Test error metrics with identical tensors."""
        if self.logger:
            self.logger.info("Testing error metrics with identical tensors")

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        true = torch.tensor([1.0, 2.0, 3.0, 4.0])

        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                num_elements, value = func(pred, true)
                self.assertEqual(num_elements, 4)

                # R2 behaves differently from error-based metrics:
                # for a perfect prediction, R2 = PEARSON = 1.0 whereas error metrics equal 0.0.
                if name == "R2" or name == "PEARSON":
                    self.assertAlmostEqual(value.item(), 1.0, places=6)
                else:
                    self.assertEqual(value.item(), 0.0)

        if self.logger:
            self.logger.info("✅ Error metrics exact match test passed")

    def test_multi_dimensional(self):
        """Test error metrics with multi-dimensional tensors."""
        if self.logger:
            self.logger.info("Testing error metrics with multi-dimensional tensors")

        pred = torch.randn(2, 3, 4, 5)  # Batch size 2, channels 3, height 4, width 5
        true = torch.randn(2, 3, 4, 5)

        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                num_elements, value = func(pred, true)
                expected = self._compute_expected(name, pred, true)

                self.assertEqual(num_elements, pred.numel())
                self.assertAlmostEqual(value.item(), expected.item(), places=6)

        if self.logger:
            self.logger.info("✅ Error metrics multi-dimensional test passed")

    def test_different_shapes(self):
        """Test error metrics with tensors of different shapes."""
        if self.logger:
            self.logger.info("Testing error metrics with different shapes")

        pred = torch.randn(2, 3, 4)
        true = torch.randn(2, 3, 4)

        # This should work fine since shapes match
        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                num_elements, value = func(pred, true)
                expected = self._compute_expected(name, pred, true)

                self.assertEqual(num_elements, 2 * 3 * 4)  # 24
                self.assertIsInstance(value, torch.Tensor)
                self.assertAlmostEqual(value.item(), expected.item(), places=6)

        # Test with mismatched shapes (should fail)
        true_wrong = torch.randn(2, 4, 3)  # Different shape

        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                with self.assertRaises(RuntimeError):
                    func(pred, true_wrong)

        if self.logger:
            self.logger.info("✅ Error metrics shape tests passed")

    def test_dtype_preservation(self):
        """Test that error metrics preserve data types."""
        if self.logger:
            self.logger.info("Testing error metrics data type preservation")

        for dtype in (torch.float32, torch.float64):
            pred = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
            true = torch.tensor([1.1, 1.9, 3.2], dtype=dtype)

            for name, func in self.metrics.items():
                with self.subTest(metric=name, dtype=dtype):
                    _, value = func(pred, true)
                    self.assertEqual(value.dtype, dtype)

        if self.logger:
            self.logger.info("✅ Error metrics dtype preservation test passed")

    def test_example_from_docstring(self):
        """Test error metrics examples from their docstrings."""
        if self.logger:
            self.logger.info("Testing error metrics docstring examples")

        pred = torch.tensor([1.0, 2.0, 3.0])
        true = torch.tensor([1.1, 1.9, 3.2])

        for name, func in self.metrics.items():
            with self.subTest(metric=name):
                num_elements, value = func(pred, true)
                expected = self._compute_expected(name, pred, true)

                self.assertEqual(num_elements, 3)
                self.assertAlmostEqual(value.item(), expected.item(), places=6)

        if self.logger:
            self.logger.info("✅ Error metrics docstring example tests passed")

    # KL is estimated via histograms (numerical approximation),
    # so no exact analytical expected value can be computed.
    def test_kl_divergence_basic(self):
        """Test KL divergence properties."""
        if self.logger:
            self.logger.info("Testing KL divergence basic properties")

        torch.manual_seed(0)

        # Identical distributions → KL ≈ 0
        true = torch.randn(1000)
        pred_same = true.clone()

        num_elements, kl_same = kl_divergence_all(pred_same, true)

        self.assertEqual(num_elements, true.numel())
        self.assertTrue(torch.isfinite(kl_same))
        self.assertAlmostEqual(kl_same.item(), 0.0, places=4)

        # Different distributions → KL > 0
        pred_shifted = true + 2.0  # shift distribution

        _, kl_diff = kl_divergence_all(pred_shifted, true)

        self.assertTrue(torch.isfinite(kl_diff))
        self.assertGreaterEqual(kl_diff.item(), 0.0)
        self.assertGreater(kl_diff.item(), kl_same.item())

        if self.logger:
            self.logger.info("✅ KL divergence basic test passed")

    def test_kl_different_shapes(self):
        """
        KL divergence should raise RuntimeError if tensor shapes differ.
        """

        if self.logger:
            self.logger.info("Testing KL divergence shape mismatch")

        pred = torch.randn(10)
        true = torch.randn(5)

        with self.assertRaises(RuntimeError):
            kl_divergence_all(pred, true)

        if self.logger:
            self.logger.info("✅ KL shape mismatch test passed")

    def test_kl_dtype_preservation(self):
        """
        Ensure KL divergence preserves the input tensor dtype.
        """

        if self.logger:
            self.logger.info("Testing KL divergence dtype preservation")

        true_f32 = torch.randn(500, dtype=torch.float32)
        pred_f32 = true_f32 + 0.1

        _, kl_f32 = kl_divergence_all(pred_f32, true_f32)
        self.assertEqual(kl_f32.dtype, torch.float32)

        true_f64 = true_f32.double()
        pred_f64 = pred_f32.double()

        _, kl_f64 = kl_divergence_all(pred_f64, true_f64)
        self.assertEqual(kl_f64.dtype, torch.float64)

        if self.logger:
            self.logger.info("✅ KL dtype preservation test passed")

    def test_kl_multi_dimensional(self):
        """
        KL divergence should correctly handle multi-dimensional tensors
        by flattening them internally.
        """

        if self.logger:
            self.logger.info("Testing KL divergence with multi-dimensional tensors")

        torch.manual_seed(0)

        pred = torch.randn(2, 3, 4, 5)
        true = torch.randn(2, 3, 4, 5)

        num_elements, kl_value = kl_divergence_all(pred, true)

        self.assertEqual(num_elements, pred.numel())
        self.assertTrue(torch.isfinite(kl_value))
        self.assertGreaterEqual(kl_value.item(), 0.0)

        if self.logger:
            self.logger.info("✅ KL multi-dimensional test passed")


class TestCRPSFunction(unittest.TestCase):
    """Unit tests for crps_ensemble_all function."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up crps_ensemble_all test fixtures")

    def test_crps_basic(self):
        """Test CRPS with simple known values."""
        if self.logger:
            self.logger.info("Testing CRPS basic functionality")

        true = torch.tensor([2.0, 2.0])
        pred_ens = torch.tensor(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ]
        )  # N_ens = 3

        num_elements, crps = crps_ensemble_all(pred_ens, true)

        # CRPS must be finite and non-negative
        self.assertEqual(num_elements, 2)
        self.assertTrue(torch.isfinite(crps))
        self.assertGreaterEqual(crps.item(), 0.0)

        if self.logger:
            self.logger.info("✅ CRPS basic test passed")

    def test_crps_zero_when_perfect_prediction(self):
        """Test CRPS is zero when all ensemble members equal truth."""
        if self.logger:
            self.logger.info("Testing CRPS perfect prediction")

        true = torch.tensor([1.0, 2.0, 3.0])
        pred_ens = torch.stack([true, true, true])  # N_ens = 3

        num_elements, crps = crps_ensemble_all(pred_ens, true)

        self.assertEqual(num_elements, 3)
        self.assertAlmostEqual(crps.item(), 0.0, places=6)

        if self.logger:
            self.logger.info("✅ CRPS perfect prediction test passed")

    def test_crps_equals_mae_for_single_member(self):
        """Test CRPS reduces to MAE when N_ens = 1."""
        if self.logger:
            self.logger.info("Testing CRPS equals MAE for single ensemble member")

        true = torch.tensor([1.0, 2.0, 3.0])
        pred = torch.tensor([1.5, 1.5, 2.5])
        pred_ens = pred.unsqueeze(0)  # [1, N_pixels]

        num_elements, crps = crps_ensemble_all(pred_ens, true)
        expected_mae = torch.mean(torch.abs(pred - true))

        self.assertEqual(num_elements, 3)
        self.assertAlmostEqual(crps.item(), expected_mae.item(), places=6)

        if self.logger:
            self.logger.info("✅ CRPS single-member equals MAE test passed")

    def test_crps_multi_dimensional_flatten(self):
        """Test CRPS with flattened multi-dimensional data."""
        if self.logger:
            self.logger.info("Testing CRPS multi-dimensional flattened input")

        torch.manual_seed(0)
        true = torch.randn(4 * 5)
        pred_ens = torch.randn(6, 4 * 5)

        num_elements, crps = crps_ensemble_all(pred_ens, true)

        self.assertEqual(num_elements, 20)
        self.assertTrue(torch.isfinite(crps))
        self.assertGreaterEqual(crps.item(), 0.0)

        if self.logger:
            self.logger.info("✅ CRPS multi-dimensional test passed")

    def test_crps_dtype_preservation(self):
        """Test CRPS preserves floating point dtype."""
        if self.logger:
            self.logger.info("Testing CRPS dtype preservation")

        true_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        pred_ens_f32 = torch.tensor([[1.5, 2.5]], dtype=torch.float32)

        _, crps_f32 = crps_ensemble_all(pred_ens_f32, true_f32)
        self.assertEqual(crps_f32.dtype, torch.float32)

        true_f64 = true_f32.double()
        pred_ens_f64 = pred_ens_f32.double()

        _, crps_f64 = crps_ensemble_all(pred_ens_f64, true_f64)
        self.assertEqual(crps_f64.dtype, torch.float64)

        if self.logger:
            self.logger.info("✅ CRPS dtype preservation test passed")


class TestDenormalizeFunction(unittest.TestCase):
    """Unit tests for denormalize function."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up denormalize test fixtures")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_denormalize_minmax(self):
        """Test denormalize with minmax normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - minmax")

        # Create test data
        data = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            vmin = 10.0
            vmax = 20.0

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "minmax", self.device)

        # Expected: data * (vmax - vmin) + vmin
        # For values [0, 0.5, 1.0] and range [10, 20]
        expected = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32).to(self.device)

        torch.testing.assert_close(result, expected)

        if self.logger:
            self.logger.info("✅ denormalize minmax test passed")

    def test_denormalize_minmax_11(self):
        """Test denormalize with minmax_11 normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - minmax_11")

        # Create test data in range [-1, 1]
        data = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            vmin = 0.0
            vmax = 100.0

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "minmax_11", self.device)

        # Expected: ((data + 1) / 2) * (vmax - vmin) + vmin
        # For values [-1, 0, 1] and range [0, 100]
        # (-1+1)/2 * 100 + 0 = 0
        # (0+1)/2 * 100 + 0 = 50
        # (1+1)/2 * 100 + 0 = 100
        expected = torch.tensor([0.0, 50.0, 100.0], dtype=torch.float32).to(self.device)

        torch.testing.assert_close(result, expected)

        if self.logger:
            self.logger.info("✅ denormalize minmax_11 test passed")

    def test_denormalize_standard(self):
        """Test denormalize with standard normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - standard")

        # Create test data (normalized, mean=0, std=1)
        data = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            vmean = 50.0
            vstd = 10.0

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "standard", self.device)

        # Expected: data * std + mean
        # For values [-2, 0, 2] with mean=50, std=10
        expected = torch.tensor([30.0, 50.0, 70.0], dtype=torch.float32).to(self.device)

        torch.testing.assert_close(result, expected)

        if self.logger:
            self.logger.info("✅ denormalize standard test passed")

    def test_denormalize_robust(self):
        """Test denormalize with robust normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - robust")

        # Create test data
        data = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            median = 50.0
            iqr = 20.0

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "robust", self.device)

        # Expected: data * iqr + median
        # For values [-1, 0, 1] with median=50, iqr=20
        expected = torch.tensor([30.0, 50.0, 70.0], dtype=torch.float32).to(self.device)

        torch.testing.assert_close(result, expected)

        if self.logger:
            self.logger.info("✅ denormalize robust test passed")

    def test_denormalize_log1p_minmax(self):
        """Test denormalize with log1p_minmax normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - log1p_minmax")

        # Normalized data
        data = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            # log1p(0) = 0
            vmin = 0.0
            vmax = torch.log1p(torch.tensor(9.0)).item()  # log1p(9) ≈ 2.3026

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "log1p_minmax", self.device)

        # Expected values:
        # z=0   → log1p(x)=0, x=0
        # z=0.5 → log1p(x) = 0.5 * 2.302585 = 1.1513, x = exp(1.151293) - 1 ≈ 2.1623
        # z=1   → log1p(x)=~2.3026, x=9
        expected = torch.expm1(data * (stats.vmax - stats.vmin) + stats.vmin)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        if self.logger:
            self.logger.info("✅ denormalize log1p_minmax test passed")

    def test_denormalize_log1p_standard(self):
        """Test denormalize with log1p_standard normalization."""
        if self.logger:
            self.logger.info("Testing denormalize - log1p_standard")

        # Normalized data
        data = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).to(self.device)

        # Create stats object
        class Stats:
            vmean = torch.log1p(torch.tensor(4.0)).item()  # log1p(4) ≈ 1.6094
            vstd = 0.5

        stats = Stats()

        # Denormalize
        result = denormalize(data, stats, "log1p_standard", self.device)

        # Expected:
        # z=-1 → log1p(x)=1.1094, x≈2.03
        # z=0  → log1p(x)=1.6094, x=4
        # z=1  → log1p(x)=2.1094, x≈7.24
        expected = torch.tensor(
            [
                torch.expm1(torch.tensor(1.6094 - 0.5)),
                4.0,
                torch.expm1(torch.tensor(1.6094 + 0.5)),
            ],
            dtype=torch.float32,
        ).to(self.device)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

        if self.logger:
            self.logger.info("✅ denormalize log1p_standard test passed")

    def test_denormalize_zero_denominator(self):
        """Test denormalize with zero denominator."""
        if self.logger:
            self.logger.info("Testing denormalize - zero denominator")

        # Test minmax with zero range
        data = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32).to(self.device)

        class StatsZeroRange:
            vmin = 10.0
            vmax = 10.0  # Zero range

        stats_zero = StatsZeroRange()
        result = denormalize(data, stats_zero, "minmax", self.device)
        expected_zero = torch.zeros_like(data).to(self.device)
        torch.testing.assert_close(result, expected_zero)

        # Test standard with zero std
        class StatsZeroStd:
            vmean = 50.0
            vstd = 0.0

        stats_zero_std = StatsZeroStd()
        result_std = denormalize(data, stats_zero_std, "standard", self.device)
        expected_zero_std = torch.zeros_like(data).to(self.device)
        torch.testing.assert_close(result_std, expected_zero_std)

        if self.logger:
            self.logger.info("✅ denormalize zero denominator test passed")

    def test_denormalize_unsupported_type(self):
        """Test denormalize with unsupported normalization type."""
        if self.logger:
            self.logger.info("Testing denormalize - unsupported type")

        data = torch.tensor([1.0], dtype=torch.float32).to(self.device)

        class Stats:
            pass

        stats = Stats()

        # Should raise ValueError for unsupported type
        with self.assertRaises(ValueError):
            denormalize(data, stats, "unsupported_type", self.device)

        if self.logger:
            self.logger.info("✅ denormalize unsupported type test passed")


class TestRunValidation(unittest.TestCase):
    """Unit tests for run_validation function focusing on return values verification."""

    def setUp(self):
        """Set up test fixtures."""

        # Create logger
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        if self.logger:
            self.logger.info("Setting up run_validation test fixtures")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_val_loss_and_metrics_across_3_batches_consistent_shape(self):
        """Verify avg_val_loss and val_metrics with 3 batches of consistent shape."""
        if self.logger:
            self.logger.info(
                "Testing avg_val_loss and val_metrics with 3 consistent batches"
            )

        # ========================================================================
        # SETUP: Debug version - track everything
        # ========================================================================

        # Create mock model
        mock_model = Mock()
        mock_loss_fn = Mock()
        mock_logger = Mock()
        mock_valid_dataset = Mock()
        mock_steps = Mock()

        # Consistent batch configuration
        batch_size = 2
        num_channels = 5
        num_vars = 2
        H, W = 8, 8

        # Create 3 batches
        batch1 = {
            "inputs": torch.randn(batch_size, num_channels, H, W).to(self.device),
            "targets": torch.randn(batch_size, num_vars, H, W).to(self.device),
            "coarse": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 10.0,
                    torch.ones(batch_size, 1, H, W) * 20.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "fine": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 12.0,
                    torch.ones(batch_size, 1, H, W) * 22.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "doy": torch.tensor([100, 150]).to(self.device),
            "hour": torch.tensor([12, 18]).to(self.device),
            "corrdinates": {
                "lat": torch.randn(batch_size, H, W).to(self.device),
                "lon": torch.randn(batch_size, H, W).to(self.device),
            },
        }

        batch2 = {
            "inputs": torch.randn(batch_size, num_channels, H, W).to(self.device),
            "targets": torch.randn(batch_size, num_vars, H, W).to(self.device),
            "coarse": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 15.0,
                    torch.ones(batch_size, 1, H, W) * 25.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "fine": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 17.0,
                    torch.ones(batch_size, 1, H, W) * 27.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "doy": torch.tensor([200, 250]).to(self.device),
            "hour": torch.tensor([6, 12]).to(self.device),
            "corrdinates": {
                "lat": torch.randn(batch_size, H, W).to(self.device),
                "lon": torch.randn(batch_size, H, W).to(self.device),
            },
        }

        batch3 = {
            "inputs": torch.randn(batch_size, num_channels, H, W).to(self.device),
            "targets": torch.randn(batch_size, num_vars, H, W).to(self.device),
            "coarse": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 20.0,
                    torch.ones(batch_size, 1, H, W) * 30.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "fine": torch.stack(
                [
                    torch.ones(batch_size, 1, H, W) * 21.0,
                    torch.ones(batch_size, 1, H, W) * 31.0,
                ],
                dim=1,
            )
            .squeeze(2)
            .to(self.device),
            "doy": torch.tensor([50, 100]).to(self.device),
            "hour": torch.tensor([20, 8]).to(self.device),
            "corrdinates": {
                "lat": torch.randn(batch_size, H, W).to(self.device),
                "lon": torch.randn(batch_size, H, W).to(self.device),
            },
        }

        batch1["inputs"][:, :num_vars] = batch1["coarse"]
        batch2["inputs"][:, :num_vars] = batch2["coarse"]
        batch3["inputs"][:, :num_vars] = batch3["coarse"]

        mock_valid_loader = [batch1, batch2, batch3]

        # Mock args
        mock_args = Mock()
        mock_args.varnames_list = ["temp", "pressure"]
        mock_args.time_normalization = "linear"
        mock_args.debug = False
        mock_args.inference_type = "direct"

        # ========================================================================
        # SETUP: Better mock tracking
        # ========================================================================

        # Track loss calls
        loss_calls = []

        def loss_fn_side_effect(model, targets, features, labels):
            # Determine which batch based on the hour values (unique per batch)
            hour_sum = (
                labels[:, 1].sum().item()
            )  # labels is [batch_size, 2] where second column is hour

            # Map hour sum to batch
            if hour_sum == (12 + 18):  # Batch 1
                loss_value = 0.35
                batch_num = 1
            elif hour_sum == (6 + 12):  # Batch 2
                loss_value = 0.85
                batch_num = 2
            else:  # Batch 3
                loss_value = 1.35
                batch_num = 3

            # Create loss tensor
            loss_tensor = torch.full_like(targets, loss_value)
            loss_calls.append((batch_num, loss_value, loss_tensor.mean().item()))

            return loss_tensor

        mock_loss_fn.side_effect = loss_fn_side_effect
        mock_loss_fn.P_mean = 0.0
        mock_loss_fn.P_std = 1.0

        # Track model calls
        model_calls = []

        def model_side_effect(x, sigma, condition_img=None, class_labels=None):
            batch_size_local = x.shape[0]

            # Simple: return based on call count
            call_num = len(model_calls)
            model_calls.append(call_num)

            if call_num == 0:  # Batch 1
                return (
                    torch.stack(
                        [
                            torch.ones(batch_size_local, 1, H, W) * 1.0,
                            torch.ones(batch_size_local, 1, H, W) * 2.0,
                        ],
                        dim=1,
                    )
                    .squeeze(2)
                    .to(self.device)
                )
            elif call_num == 1:  # Batch 2
                return (
                    torch.stack(
                        [
                            torch.ones(batch_size_local, 1, H, W) * 1.5,
                            torch.ones(batch_size_local, 1, H, W) * 2.5,
                        ],
                        dim=1,
                    )
                    .squeeze(2)
                    .to(self.device)
                )
            else:  # Batch 3
                return (
                    torch.stack(
                        [
                            torch.ones(batch_size_local, 1, H, W) * 0.5,
                            torch.ones(batch_size_local, 1, H, W) * 1.5,
                        ],
                        dim=1,
                    )
                    .squeeze(2)
                    .to(self.device)
                )

        mock_model.side_effect = model_side_effect

        # Mock normalization
        mock_norm_mapping = {}
        mock_normalization_type = {}
        mock_index_mapping = {}

        class MockStats:
            vmin = 0.0
            vmax = 1.0
            vmean = 0.0
            vstd = 1.0

        mock_norm_mapping["temp_fine"] = MockStats()
        mock_norm_mapping["pressure_fine"] = MockStats()
        mock_normalization_type["temp"] = "minmax"
        mock_normalization_type["pressure"] = "minmax"
        mock_index_mapping["temp"] = 0
        mock_index_mapping["pressure"] = 1

        # ========================================================================
        # EXECUTE: Run validation
        # ========================================================================

        with patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x):
            with patch("torch.amp.autocast"):
                avg_val_loss, val_metrics = run_validation(
                    model=mock_model,
                    valid_dataset=mock_valid_dataset,
                    valid_loader=mock_valid_loader,
                    loss_fn=mock_loss_fn,
                    norm_mapping=mock_norm_mapping,
                    normalization_type=mock_normalization_type,
                    index_mapping=mock_index_mapping,
                    args=mock_args,
                    steps=mock_steps,
                    device=self.device,
                    logger=mock_logger,
                    epoch=1,
                    writer=None,
                    plot_every_n_epochs=None,
                    paths=None,
                )

        # ========================================================================
        # DEBUG: Print what happened
        # ========================================================================

        if self.logger:
            self.logger.info(f"Loss calls: {loss_calls}")
            self.logger.info(f"Model calls: {model_calls}")
            self.logger.info(f"avg_val_loss = {avg_val_loss}")

            # Calculate expected based on actual loss values
            if loss_calls:
                actual_losses = [loss_entry[1] for loss_entry in loss_calls]
                expected = sum(
                    loss_value * batch_size for loss_value in actual_losses
                ) / (len(actual_losses) * batch_size)
                self.logger.info(f"Actual losses: {actual_losses}")
                self.logger.info(f"Expected avg: {expected}")

        # ========================================================================
        # VERIFICATION 1: avg_val_loss calculation
        # ========================================================================

        # Get the actual loss values that were used
        actual_loss_values = (
            [loss_entry[1] for loss_entry in loss_calls]
            if loss_calls
            else [0.35, 0.85, 1.35]
        )

        # Calculate expected based on ACTUAL values
        expected_avg_loss = sum(
            loss_value * batch_size for loss_value in actual_loss_values
        ) / (len(actual_loss_values) * batch_size)

        self.assertAlmostEqual(
            avg_val_loss,
            expected_avg_loss,
            places=5,
            msg=f"Expected avg_val_loss={expected_avg_loss:.5f} (based on losses {actual_loss_values}), got {avg_val_loss:.5f}",
        )

        if self.logger:
            self.logger.info(
                f"✅ avg_val_loss verified: {avg_val_loss:.5f} (expected: {expected_avg_loss:.5f})"
            )

        # ========================================================================
        # VERIFICATION 2: Compute expected MAE values manually
        # ========================================================================

        # Each batch has 2 samples × 8×8 elements = 128 elements per batch per variable
        # Total elements per variable: 3 batches × 128 = 384

        # TEMP variable calculations:
        # Batch 1: coarse=10, residual=1.0, pred=11.0, fine=12.0 → MAE=|11-12|=1.0
        # Batch 2: coarse=15, residual=1.5, pred=16.5, fine=17.0 → MAE=|16.5-17|=0.5
        # Batch 3: coarse=20, residual=0.5, pred=20.5, fine=21.0 → MAE=|20.5-21|=0.5

        # Element-weighted temp pred MAE:
        # (1.0×128 + 0.5×128 + 0.5×128) / 384 = (128 + 64 + 64)/384 = 256/384 = 0.6666667

        expected_temp_pred_mae = 256 / 384

        # Temp coarse MAE:
        # Batch 1: |10-12|=2.0, Batch 2: |15-17|=2.0, Batch 3: |20-21|=1.0
        # (2.0×128 + 2.0×128 + 1.0×128)/384 = (256 + 256 + 128)/384 = 640/384 = 1.6666667

        expected_temp_coarse_mae = 640 / 384

        # PRESSURE variable calculations:
        # Batch 1: coarse=20, residual=2.0, pred=22.0, fine=22.0 → MAE=|22-22|=0.0
        # Batch 2: coarse=25, residual=2.5, pred=27.5, fine=27.0 → MAE=|27.5-27|=0.5
        # Batch 3: coarse=30, residual=1.5, pred=31.5, fine=31.0 → MAE=|31.5-31|=0.5

        # Element-weighted pressure pred MAE:
        # (0.0×128 + 0.5×128 + 0.5×128)/384 = (0 + 64 + 64)/384 = 128/384 = 0.3333333

        expected_pressure_pred_mae = 128 / 384

        # Pressure coarse MAE (same as temp):
        # (2.0×128 + 2.0×128 + 1.0×128)/384 = 640/384 = 1.6666667

        expected_pressure_coarse_mae = 640 / 384

        # ========================================================================
        # VERIFICATION 3: Verify per-variable MAE values
        # ========================================================================

        # Get actual values
        actual_temp_pred = val_metrics["temp_pred_vs_fine_MAE"].getmean()
        actual_temp_coarse = val_metrics["temp_coarse_vs_fine_MAE"].getmean()
        actual_pressure_pred = val_metrics["pressure_pred_vs_fine_MAE"].getmean()
        actual_pressure_coarse = val_metrics["pressure_coarse_vs_fine_MAE"].getmean()

        # Verify temp MAE
        self.assertAlmostEqual(
            actual_temp_pred,
            expected_temp_pred_mae,
            places=5,
            msg=f"Temp pred MAE: expected {expected_temp_pred_mae:.5f}, got {actual_temp_pred:.5f}",
        )
        self.assertAlmostEqual(
            actual_temp_coarse,
            expected_temp_coarse_mae,
            places=5,
            msg=f"Temp coarse MAE: expected {expected_temp_coarse_mae:.5f}, got {actual_temp_coarse:.5f}",
        )

        # Verify pressure MAE
        self.assertAlmostEqual(
            actual_pressure_pred,
            expected_pressure_pred_mae,
            places=5,
            msg=f"Pressure pred MAE: expected {expected_pressure_pred_mae:.5f}, got {actual_pressure_pred:.5f}",
        )
        self.assertAlmostEqual(
            actual_pressure_coarse,
            expected_pressure_coarse_mae,
            places=5,
            msg=f"Pressure coarse MAE: expected {expected_pressure_coarse_mae:.5f}, got {actual_pressure_coarse:.5f}",
        )

        if self.logger:
            self.logger.info("✅ Per-variable MAE verified:")
            self.logger.info(
                f" └── temp_pred: {actual_temp_pred:.5f} (expected: {expected_temp_pred_mae:.5f})"
            )
            self.logger.info(
                f" └── temp_coarse: {actual_temp_coarse:.5f} (expected: {expected_temp_coarse_mae:.5f})"
            )
            self.logger.info(
                f" └── pressure_pred: {actual_pressure_pred:.5f} (expected: {expected_pressure_pred_mae:.5f})"
            )
            self.logger.info(
                f" └── pressure_coarse: {actual_pressure_coarse:.5f} (expected: {expected_pressure_coarse_mae:.5f})"
            )

        # ========================================================================
        # VERIFICATION 4: Compute and verify average metrics
        # ========================================================================

        # Average metrics are computed per-batch (not weighted by elements)
        # Let's compute expected batch-level averages:

        # Batch 1 averages:
        # - pred: (temp_pred=1.0, pressure_pred=0.0) → avg = (1.0+0.0)/2 = 0.5
        # - coarse: (temp_coarse=2.0, pressure_coarse=2.0) → avg = (2.0+2.0)/2 = 2.0

        # Batch 2 averages:
        # - pred: (0.5 + 0.5)/2 = 0.5
        # - coarse: (2.0 + 2.0)/2 = 2.0

        # Batch 3 averages:
        # - pred: (0.5 + 0.5)/2 = 0.5
        # - coarse: (1.0 + 1.0)/2 = 1.0

        # Overall averages (simple mean across batches):
        expected_avg_pred = (0.5 + 0.5 + 0.5) / 3  # = 0.5
        expected_avg_coarse = (2.0 + 2.0 + 1.0) / 3  # = 1.6666667

        actual_avg_pred = val_metrics["average_pred_vs_fine_MAE"].getmean()
        actual_avg_coarse = val_metrics["average_coarse_vs_fine_MAE"].getmean()

        self.assertAlmostEqual(
            actual_avg_pred,
            expected_avg_pred,
            places=5,
            msg=f"Avg pred MAE: expected {expected_avg_pred:.5f}, got {actual_avg_pred:.5f}",
        )
        self.assertAlmostEqual(
            actual_avg_coarse,
            expected_avg_coarse,
            places=5,
            msg=f"Avg coarse MAE: expected {expected_avg_coarse:.5f}, got {actual_avg_coarse:.5f}",
        )

        if self.logger:
            self.logger.info("✅ Average metrics verified:")
            self.logger.info(
                f" └── avg_pred: {actual_avg_pred:.5f} (expected: {expected_avg_pred:.5f})"
            )
            self.logger.info(
                f" └── avg_coarse: {actual_avg_coarse:.5f} (expected: {expected_avg_coarse:.5f})"
            )

        # ========================================================================
        # VERIFICATION 5: Verify MetricTracker counts
        # ========================================================================

        # Per-variable trackers should count total elements
        # 3 batches × 2 samples × 8×8 elements = 384 elements per variable
        total_elements_per_var = 3 * batch_size * H * W  # 384

        self.assertEqual(
            val_metrics["temp_pred_vs_fine_MAE"].count,
            total_elements_per_var,
            f"Temp pred tracker count should be {total_elements_per_var}, got {val_metrics['temp_pred_vs_fine_MAE'].count}",
        )
        self.assertEqual(
            val_metrics["temp_coarse_vs_fine_MAE"].count,
            total_elements_per_var,
            f"Temp coarse tracker count should be {total_elements_per_var}, got {val_metrics['temp_coarse_vs_fine_MAE'].count}",
        )
        self.assertEqual(
            val_metrics["pressure_pred_vs_fine_MAE"].count,
            total_elements_per_var,
            f"Pressure pred tracker count should be {total_elements_per_var}, got {val_metrics['pressure_pred_vs_fine_MAE'].count}",
        )
        self.assertEqual(
            val_metrics["pressure_coarse_vs_fine_MAE"].count,
            total_elements_per_var,
            f"Pressure coarse tracker count should be {total_elements_per_var}, got {val_metrics['pressure_coarse_vs_fine_MAE'].count}",
        )

        # Average trackers should count number of batches
        self.assertEqual(
            val_metrics["average_pred_vs_fine_MAE"].count,
            3,
            f"Avg pred tracker count should be 3, got {val_metrics['average_pred_vs_fine_MAE'].count}",
        )
        self.assertEqual(
            val_metrics["average_coarse_vs_fine_MAE"].count,
            3,
            f"Avg coarse tracker count should be 3, got {val_metrics['average_coarse_vs_fine_MAE'].count}",
        )

        if self.logger:
            self.logger.info(
                f"✅ Tracker counts verified: per-var={total_elements_per_var}, avg=3"
            )

        # ========================================================================
        # VERIFICATION 6: Verify all MetricTracker values are positive
        # ========================================================================

        for key, tracker in val_metrics.items():
            if tracker.count > 0:
                value = tracker.getmean()
                # R2 can be negative when the model performs worse than predicting the mean.
                # # PEARSON can be negative or undefined (NaN when variance is zero).
                # Therefore, we only enforce non-negativity for error-based metrics.
                if "R2" not in key and "PEARSON" not in key:
                    self.assertGreaterEqual(
                        value, 0.0, f"{key} should be non-negative, got {value}"
                    )

        if self.logger:
            self.logger.info(
                "✅ All error-based metric values (except R2 and PEARSON) are non-negative"
            )

        # ========================================================================
        # VERIFICATION 7: Verify function call counts
        # ========================================================================

        self.assertEqual(
            mock_loss_fn.call_count,
            3,
            f"loss_fn should be called 3 times, got {mock_loss_fn.call_count}",
        )
        self.assertEqual(
            mock_model.call_count,
            3,
            f"model should be called 3 times, got {mock_model.call_count}",
        )

        if self.logger:
            self.logger.info(
                f"✅ Function calls verified: loss_fn={mock_loss_fn.call_count}, model={mock_model.call_count}"
            )

        # ========================================================================
        # VERIFICATION 8: Final summary
        # ========================================================================

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("COMPREHENSIVE VERIFICATION SUMMARY:")
            self.logger.info("=" * 60)
            self.logger.info(f"avg_val_loss: {avg_val_loss:.5f}")
            self.logger.info(f"temp_pred_mae: {actual_temp_pred:.5f}")
            self.logger.info(f"temp_coarse_mae: {actual_temp_coarse:.5f}")
            self.logger.info(f"pressure_pred_mae: {actual_pressure_pred:.5f}")
            self.logger.info(f"pressure_coarse_mae: {actual_pressure_coarse:.5f}")
            self.logger.info(f"avg_pred_mae: {actual_avg_pred:.5f}")
            self.logger.info(f"avg_coarse_mae: {actual_avg_coarse:.5f}")
            self.logger.info(f"Tracker counts: per-var={total_elements_per_var}, avg=3")
            self.logger.info(
                f"Function calls: loss_fn={mock_loss_fn.call_count}, model={mock_model.call_count}"
            )
            self.logger.info("=" * 60)
            self.logger.info("✅ ALL VERIFICATIONS PASSED!")

    def test_crps_zero_when_predictions_equal_fine(self):
        """
        Verify that CRPS is zero when all ensemble predictions
        exactly match the fine target.
        """
        if self.logger:
            self.logger.info("Testing CRPS = 0 when predictions equal fine")

        mock_model = Mock()
        mock_loss_fn = Mock()
        mock_logger = Mock()
        mock_steps = Mock()

        batch_size = 2
        num_channels = 5
        num_vars = 1
        H, W = 8, 8

        fine = torch.ones(batch_size, num_vars, H, W, device=self.device)
        coarse = torch.zeros_like(fine)

        batch = {
            "inputs": torch.zeros(batch_size, num_channels, H, W, device=self.device),
            "targets": torch.zeros(batch_size, num_vars, H, W, device=self.device),
            "coarse": coarse,
            "fine": fine,
            "doy": torch.tensor([100, 200], device=self.device),
            "hour": torch.tensor([12, 18], device=self.device),
            "corrdinates": {
                "lat": torch.zeros(batch_size, H, device=self.device),
                "lon": torch.zeros(batch_size, W, device=self.device),
            },
        }

        mock_valid_loader = [batch]

        # Sampler always returns perfect residuals
        def mock_sampler(*args, **kwargs):
            return fine - coarse  # exact residuals

        mock_loss_fn.return_value = torch.zeros_like(batch["targets"])
        mock_loss_fn.P_mean = 0.0
        mock_loss_fn.P_std = 1.0

        class MockStats:
            vmin = 0.0
            vmax = 1.0

        # mock_norm_mapping = {"var_residual": MockStats()}
        mock_norm_mapping = {"var_fine": MockStats()}
        mock_normalization_type = {"var": "minmax"}
        mock_index_mapping = {"var": 0}

        mock_args = Mock()
        mock_args.varnames_list = ["var"]
        mock_args.time_normalization = "linear"
        mock_args.inference_type = "sampler"
        mock_args.debug = False

        with patch("tqdm.tqdm", side_effect=lambda x, **kwargs: x):
            with patch("IPSL_AID.evaluater.sampler", side_effect=mock_sampler):
                with patch("torch.amp.autocast"):
                    _, val_metrics = run_validation(
                        model=mock_model,
                        valid_dataset=Mock(),
                        valid_loader=mock_valid_loader,
                        loss_fn=mock_loss_fn,
                        norm_mapping=mock_norm_mapping,
                        normalization_type=mock_normalization_type,
                        index_mapping=mock_index_mapping,
                        args=mock_args,
                        steps=mock_steps,
                        device=self.device,
                        logger=mock_logger,
                        epoch=0,
                        compute_crps=True,
                        crps_batch_size=1,
                        crps_ensemble_size=5,
                        writer=None,
                        plot_every_n_epochs=None,
                        paths=None,
                    )

        for var_name in mock_args.varnames_list:
            key = f"{var_name}_pred_vs_fine_CRPS"
            self.assertIn(key, val_metrics)

            crps_value = val_metrics[key].getmean()
            self.assertAlmostEqual(crps_value, 0.0, places=6)

        if self.logger:
            self.logger.info("✅ CRPS test passed (predictions == fine, CRPS = 0)")

    def test_generate_residuals_matches_fine(self):
        """
        Final prediction (coarse + residuals) should exactly
        match fine when residuals = fine - coarse.
        """
        batch_size = 2
        H, W = 4, 4
        num_vars = 2
        device = self.device

        fine = torch.ones(batch_size, num_vars, H, W, device=device)
        coarse = torch.zeros_like(fine)
        targets = torch.zeros_like(fine)

        features = torch.zeros(batch_size, 1, H, W, device=device)
        labels = torch.zeros(batch_size, 2, device=device)

        # Sampler always returns perfect residuals
        def mock_sampler(*args, **kwargs):
            return fine - coarse

        mock_model = Mock()
        mock_loss_fn = Mock()
        mock_loss_fn.P_mean = 0.0
        mock_loss_fn.P_std = 1.0

        args = Mock()
        args.varnames_list = ["var1", "var2"]
        args.debug = False

        with patch("IPSL_AID.evaluater.sampler", side_effect=mock_sampler):
            generated_residuals = generate_residuals_norm(
                model=mock_model,
                features=features,
                labels=labels,
                targets=targets,
                loss_fn=mock_loss_fn,
                args=args,
                device=device,
                logger=Mock(),
                inference_type="sampler",
            )

        # Shape check (residuals)
        self.assertEqual(generated_residuals.shape, fine.shape)

        # Reconstruct final prediction
        final_pred = coarse + generated_residuals

        # Exact reconstruction
        self.assertTrue(torch.allclose(final_pred, fine, atol=1e-6))

        if self.logger:
            self.logger.info("✅ The reconstructed prediction matches the fine data")

    def tearDown(self):
        """Clean up after tests."""
        if self.logger:
            self.logger.info("Evaluater tests completed successfully")


def run_tests():
    """Run all Evaluater tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetricTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestCRPSFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestDenormalizeFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestRunValidation))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
