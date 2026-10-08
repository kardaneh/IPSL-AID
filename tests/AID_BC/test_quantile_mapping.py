# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.quantile_mapping import QM

# python -m unittest tests.test_quantile_mapping


# ============================================================================
# Unit Tests for QM
# ============================================================================


class TestQuantileMapping(unittest.TestCase):
    """Unit tests for empirical Quantile Mapping."""

    def setUp(self):
        """Create a test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_empirical_quantiles(self):
        """
        Test the empirical probability grid.

        For five sorted observations, the expected probabilities are:

        [1/5, 2/5, 3/5, 4/5, 5/5]
        """
        self.logger.info("Testing empirical quantile probabilities")

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        probabilities = QM._empirical_quantiles(values)

        np.testing.assert_allclose(
            probabilities,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Empirical quantile probabilities test passed")

    def test_empirical_mapping(self):
        """
        Test empirical Quantile Mapping.

        Biased sample:

        X = [0, 10, 20, ..., 90]

        Reference sample:

        Y = [100, 200, 300, ..., 1000]

        Both distributions use the same empirical probability grid.
        Therefore:

        X = 5   -> Y = 150
        X = 10  -> Y = 200
        X = 15  -> Y = 250
        """
        self.logger.info("Testing empirical Quantile Mapping")

        biased = np.arange(
            0.0,
            100.0,
            10.0,
        )

        reference = np.arange(
            100.0,
            1100.0,
            100.0,
        )

        values = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        expected = np.array([150.0, 200.0, 250.0, 300.0, 350.0])

        model = QM()

        model.fit(
            Y0=reference,
            X0=biased,
        )

        corrected = model.predict(values)

        self.assertEqual(
            corrected.shape,
            (values.size, 1),
        )

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Empirical Quantile Mapping test passed")

    def test_two_features(self):
        """
        Test independent Quantile Mapping of two features.

        Feature 1:

        X = [0, 10, 20, 30]
        Y = [100, 200, 300, 400]

        Feature 2:

        X = [0, 1, 2, 3]
        Y = [10, 20, 30, 40]
        """
        self.logger.info("Testing two-feature Quantile Mapping")

        biased = np.array(
            [
                [0.0, 0.0],
                [10.0, 1.0],
                [20.0, 2.0],
                [30.0, 3.0],
            ]
        )

        reference = np.array(
            [
                [100.0, 10.0],
                [200.0, 20.0],
                [300.0, 30.0],
                [400.0, 40.0],
            ]
        )

        values = np.array(
            [
                [5.0, 0.5],
                [15.0, 1.5],
                [25.0, 2.5],
            ]
        )

        expected = np.array(
            [
                [150.0, 15.0],
                [250.0, 25.0],
                [350.0, 35.0],
            ]
        )

        model = QM()

        model.fit(
            Y0=reference,
            X0=biased,
        )

        corrected = model.predict(values)

        self.assertEqual(
            corrected.shape,
            expected.shape,
        )

        np.testing.assert_allclose(
            corrected,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Two-feature Quantile Mapping test passed")

    def test_values_outside_fitted_range(self):
        """
        Test values outside the fitted biased range.

        Values below or above the fitted biased distribution are clipped
        to the minimum or maximum reference quantile.
        """
        self.logger.info("Testing Quantile Mapping outside fitted range")

        biased = np.array([0.0, 10.0, 20.0, 30.0])
        reference = np.array([100.0, 200.0, 300.0, 400.0])

        values = np.array([-10.0, 40.0])

        expected = np.array([100.0, 400.0])

        model = QM()

        model.fit(
            Y0=reference,
            X0=biased,
        )

        corrected = model.predict(values)

        np.testing.assert_allclose(
            corrected[:, 0],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Quantile Mapping range-clipping test passed")

    def test_predict_before_fit(self):
        """Test that predict() cannot be called before fit()."""
        self.logger.info("Testing predict() before fit()")

        model = QM()

        with self.assertRaises(RuntimeError):
            model.predict(np.array([0.0, 1.0, 2.0]))

        self.logger.info("✅ predict() before fit() test passed")

    def test_fit_feature_mismatch(self):
        """
        Test that reference and biased datasets must have the same
        number of features.
        """
        self.logger.info("Testing feature mismatch during fit()")

        reference = np.zeros((10, 2))
        biased = np.zeros((10, 1))

        model = QM()

        with self.assertRaises(ValueError):
            model.fit(
                Y0=reference,
                X0=biased,
            )

        self.logger.info("✅ fit() feature mismatch test passed")

    def test_predict_feature_mismatch(self):
        """
        Test that prediction data must have the same number of features
        as the fitted data.
        """
        self.logger.info("Testing feature mismatch during predict()")

        reference = np.zeros((10, 2))
        biased = np.zeros((10, 2))

        model = QM()

        model.fit(
            Y0=reference,
            X0=biased,
        )

        values = np.zeros((5, 3))

        with self.assertRaises(ValueError):
            model.predict(values)

        self.logger.info("✅ predict() feature mismatch test passed")


def run_tests():
    """Run all Quantile Mapping tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestQuantileMapping))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
