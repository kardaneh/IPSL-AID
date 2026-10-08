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
from types import SimpleNamespace
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.bias_corrector import (
    BiasCorrector,
    OptimalTransportCorrector,
    QuantileMappingCorrector,
    create_bias_corrector,
)

# python -m unittest tests.test_bias_corrector


# ============================================================================
# Test utilities
# ============================================================================


def create_training_arrays():
    """Create small deterministic training arrays."""

    reference = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ],
        dtype=np.float32,
    )

    biased = np.array(
        [
            [1.0, 12.0],
            [2.0, 13.0],
            [3.0, 14.0],
            [4.0, 15.0],
        ],
        dtype=np.float32,
    )

    return reference, biased


# ============================================================================
# Unit tests for BiasCorrector validation
# ============================================================================


class TestBiasCorrector(unittest.TestCase):
    """Unit tests for common bias-corrector validation."""

    def setUp(self):
        """Create deterministic training arrays."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.reference, self.biased = create_training_arrays()

    def test_validate_fit_arrays(self):
        """Test valid training arrays."""
        self.logger.info("Testing valid BiasCorrector training arrays")

        BiasCorrector.validate_fit_arrays(
            reference=self.reference,
            biased=self.biased,
        )

        self.logger.info("✅ Valid BiasCorrector training arrays test passed")

    def test_validate_fit_arrays_rejects_invalid_input(self):
        """Test rejection of incompatible or non-finite training data."""
        self.logger.info("Testing invalid BiasCorrector training arrays")

        with self.assertRaises(ValueError):
            BiasCorrector.validate_fit_arrays(
                reference=self.reference,
                biased=np.ones((4, 3)),
            )

        bad_reference = self.reference.copy()
        bad_reference[0, 0] = np.nan

        with self.assertRaises(ValueError):
            BiasCorrector.validate_fit_arrays(
                reference=bad_reference,
                biased=self.biased,
            )

        self.logger.info("✅ Invalid BiasCorrector training arrays test passed")


# ============================================================================
# Unit tests for QuantileMappingCorrector
# ============================================================================


class TestQuantileMappingCorrector(unittest.TestCase):
    """Unit tests for the Quantile Mapping wrapper."""

    def setUp(self):
        """Create deterministic QM arrays."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.reference, self.biased = create_training_arrays()

        self.application = np.array(
            [
                [1.5, 12.5],
                [2.5, 13.5],
            ],
            dtype=np.float64,
        )

    @patch("AID_BC.bias_corrector.QM")
    def test_fit_and_transform(self, mock_qm_class):
        """Test Quantile Mapping fitting and transformation."""
        self.logger.info("Testing Quantile Mapping fitting and transformation")

        mock_model = mock_qm_class.return_value

        expected = np.array(
            [
                [0.5, 10.5],
                [1.5, 11.5],
            ],
            dtype=np.float32,
        )

        mock_model.predict.return_value = expected

        corrector = QuantileMappingCorrector()

        corrector.fit(
            reference=self.reference,
            biased=self.biased,
        )

        corrected = corrector.transform(self.application)

        self.assertTrue(corrector.is_fitted)
        self.assertEqual(corrector.n_features, 2)
        self.assertEqual(corrected.dtype, np.float32)

        np.testing.assert_allclose(
            corrected,
            expected,
        )

        mock_model.fit.assert_called_once()
        mock_model.predict.assert_called_once()

        self.logger.info("✅ Quantile Mapping fitting and transformation test passed")

    def test_transform_before_fit(self):
        """Test rejection of transformation before fitting."""
        self.logger.info("Testing Quantile Mapping transformation before fitting")

        corrector = QuantileMappingCorrector()

        with self.assertRaises(RuntimeError):
            corrector.transform(self.application)

        self.logger.info("✅ Quantile Mapping transformation-before-fit test passed")


# ============================================================================
# Unit tests for OptimalTransportCorrector
# ============================================================================


class TestOptimalTransportCorrector(unittest.TestCase):
    """Unit tests for the Optimal Transport wrapper."""

    def setUp(self):
        """Create deterministic OT arrays."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.reference, self.biased = create_training_arrays()

        self.application = np.array(
            [
                [2.0, 13.0],
                [3.0, 14.0],
                [4.0, 15.0],
            ],
            dtype=np.float64,
        )

    def test_initialization_and_validation(self):
        """Test OT configuration and invalid parameters."""
        self.logger.info("Testing Optimal Transport initialization and validation")

        corrector = OptimalTransportCorrector(
            epsilon=100.0,
            num_iterations=500,
            threshold=0.01,
            batch_size=2,
            dtype="float64",
        )

        self.assertEqual(corrector.epsilon, 100.0)
        self.assertEqual(corrector.num_iterations, 500)
        self.assertEqual(corrector.batch_size, 2)
        self.assertEqual(corrector.numpy_dtype, np.float64)
        self.assertFalse(corrector.is_fitted)

        with self.assertRaises(ValueError):
            OptimalTransportCorrector(epsilon=0.0)

        with self.assertRaises(ValueError):
            OptimalTransportCorrector(dtype="float16")

        self.logger.info(
            "✅ Optimal Transport initialization and validation test passed"
        )

    def test_normalization(self):
        """Test shared normalization and inverse transformation."""
        self.logger.info("Testing Optimal Transport normalization")

        corrector = OptimalTransportCorrector(
            dtype="float64",
            normalize=True,
        )

        reference = self.reference.astype(np.float64)
        biased = self.biased.astype(np.float64)

        corrector._fit_normalization(
            reference=reference,
            biased=biased,
        )

        combined = np.concatenate(
            [reference, biased],
            axis=0,
        )

        np.testing.assert_allclose(
            corrector.center,
            combined.mean(axis=0),
        )

        normalized = corrector._normalize(self.application)
        restored = corrector._denormalize(normalized)

        np.testing.assert_allclose(
            restored,
            self.application,
        )

        self.logger.info("✅ Optimal Transport normalization test passed")

    @patch("AID_BC.bias_corrector.OptimalTransport")
    def test_fit(self, mock_solver_class):
        """Test successful OT fitting without running Sinkhorn."""
        self.logger.info("Testing Optimal Transport fitting")
        mock_solver = mock_solver_class.return_value

        output = SimpleNamespace(
            converged=jnp.asarray(True),
            num_iterations=jnp.asarray(12),
            reg_ot_cost=jnp.asarray(1.25),
            gv=jnp.array(
                [0.1, 0.2, 0.3, 0.4],
                dtype=jnp.float64,
            ),
        )

        mock_solver.return_value = output
        mock_solver.transport_fn.return_value = lambda x: x

        corrector = OptimalTransportCorrector(
            epsilon=100.0,
            num_iterations=500,
            threshold=0.01,
            dtype="float64",
        )

        corrector.fit(
            reference=self.reference,
            biased=self.biased,
        )

        self.assertTrue(corrector.is_fitted)
        self.assertTrue(corrector.converged)
        self.assertEqual(corrector.fitted_iterations, 12)
        self.assertAlmostEqual(corrector.regularized_ot_cost, 1.25)

        mock_solver.transport_fn.assert_called_once()

        self.logger.info(
            "✅ Optimal Transport fitting test passed - "
            f"iterations={corrector.fitted_iterations}, "
            f"cost={corrector.regularized_ot_cost}"
        )

    @patch("AID_BC.bias_corrector.OptimalTransport")
    def test_fit_rejects_non_convergence(self, mock_solver_class):
        """Test rejection when Sinkhorn does not converge."""
        self.logger.info("Testing Optimal Transport non-convergence handling")

        mock_solver = mock_solver_class.return_value

        mock_solver.return_value = SimpleNamespace(
            converged=jnp.asarray(False),
            num_iterations=jnp.asarray(50),
            reg_ot_cost=jnp.asarray(2.0),
            gv=jnp.zeros(4),
        )

        corrector = OptimalTransportCorrector(
            num_iterations=50,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Sinkhorn did not converge",
        ):
            corrector.fit(
                reference=self.reference,
                biased=self.biased,
            )

        self.logger.info("✅ Optimal Transport non-convergence handling test passed")

    def test_transform_batches(self):
        """Test batched OT transformation and output dtype."""
        self.logger.info("Testing Optimal Transport batched transformation")

        corrector = OptimalTransportCorrector(
            batch_size=2,
            dtype="float64",
            normalize=False,
        )

        corrector.is_fitted = True
        corrector.n_features = 2
        corrector.center = np.zeros(2)
        corrector.scale = np.ones(2)

        batch_sizes = []

        def transport_function(batch):
            batch_sizes.append(batch.shape[0])
            return batch + 1.0

        corrector.transport_function = transport_function

        corrected = corrector.transform(self.application)

        np.testing.assert_allclose(
            corrected,
            self.application + 1.0,
        )

        self.assertEqual(
            batch_sizes,
            [2, 1],
        )

        self.assertEqual(
            corrected.dtype,
            np.float32,
        )

        self.logger.info(
            "✅ Optimal Transport batched transformation test passed - "
            f"batches={batch_sizes}"
        )


# ============================================================================
# Unit tests for create_bias_corrector
# ============================================================================


class TestCreateBiasCorrector(unittest.TestCase):
    """Unit tests for the bias-corrector factory."""

    def setUp(self):
        """Create test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_create_correctors(self):
        """Test creation of QM and OT correctors."""
        self.logger.info("Testing bias-corrector factory")

        qm = create_bias_corrector("qm")

        ot = create_bias_corrector(
            "ot",
            ot_epsilon=50.0,
            ot_batch_size=4,
            ot_dtype="float32",
        )

        self.assertIsInstance(
            qm,
            QuantileMappingCorrector,
        )

        self.assertIsInstance(
            ot,
            OptimalTransportCorrector,
        )

        self.assertEqual(
            ot.epsilon,
            50.0,
        )

        self.assertEqual(
            ot.batch_size,
            4,
        )

        self.logger.info("✅ Bias-corrector factory test passed - created QM and OT")

    def test_create_rejects_unknown_method(self):
        """Test rejection of an unknown correction method."""
        self.logger.info("Testing rejection of unknown bias-correction method")

        with self.assertRaises(ValueError):
            create_bias_corrector("invalid")

        self.logger.info("✅ Unknown bias-correction method rejection test passed")


# ============================================================================
# Test runner
# ============================================================================


def run_tests():
    """Run all bias-corrector tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBiasCorrector))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantileMappingCorrector))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalTransportCorrector))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateBiasCorrector))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
