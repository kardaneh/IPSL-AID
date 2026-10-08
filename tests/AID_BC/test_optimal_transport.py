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

# Force CPU execution for lightweight and reproducible unit tests.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from AID_BC.logger import Logger
from AID_BC.optimal_transport import (
    OptimalTransport,
    TransportSolution,
    _squared_euclidean_cost,
)


# python -m unittest tests.test_optimal_transport


# ============================================================================
# Unit Tests for TransportSolution
# ============================================================================


class TestTransportSolution(unittest.TestCase):
    """Unit tests for TransportSolution."""

    def setUp(self):
        """Create a test logger."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

    def test_transport_plan(self):
        """Test transport-plan construction and epsilon validation."""
        self.logger.info("Testing TransportSolution  transport plan")

        output = TransportSolution(
            potentials=(
                jnp.zeros(2),
                jnp.zeros(2),
            ),
            cost_matrix=jnp.zeros((2, 2)),
            epsilon=1.0,
            reg_ot_cost=jnp.asarray(0.0),
            threshold=1e-6,
            converged=jnp.asarray(True),
            num_iterations=jnp.asarray(1),
        )

        # With zero costs and potentials, every plan entry equals one.
        expected = np.ones((2, 2))

        np.testing.assert_allclose(
            np.asarray(output.transport_plan),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ TransportSolution transport-plan test passed")


# ============================================================================
# Unit Tests for OptimalTransport
# ============================================================================


class TestOptimalTransport(unittest.TestCase):
    """Unit tests for the Sinkhorn optimal transport solver."""

    def setUp(self):
        """Create a test logger and a small optimal transport solver."""
        self.logger = Logger(
            console_output=True,
            file_output=False,
            pretty_print=True,
            record=False,
        )

        self.solver = OptimalTransport(
            epsilon=1.0,
            num_iterations=100,
            threshold=1e-6,
        )

    def test_compute_cost(self):
        """Test the pairwise squared-Euclidean distance matrix."""
        self.logger.info("Testing squared-Euclidean cost matrix")

        x = jnp.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
            ]
        )

        y = jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
            ]
        )

        expected = np.array(
            [
                [0.0, 4.0],
                [5.0, 5.0],
            ]
        )

        cost = _squared_euclidean_cost(x, y)

        np.testing.assert_allclose(
            np.asarray(cost),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Squared-Euclidean cost-matrix test passed")

    def test_invalid_solver_parameters(self):
        """Test validation of optimal transport solver parameters."""
        self.logger.info("Testing optimal transport parameter validation")

        with self.assertRaises(ValueError):
            OptimalTransport(epsilon=0.0)

        with self.assertRaises(ValueError):
            OptimalTransport(epsilon=-1.0)

        with self.assertRaises(ValueError):
            OptimalTransport(epsilon=1.0, num_iterations=0)

        with self.assertRaises(ValueError):
            OptimalTransport(epsilon=1.0, threshold=-1.0)

        with self.assertRaises(ValueError):
            OptimalTransport(epsilon=1.0, eps_marginal=-1.0)

        self.logger.info("✅ Optimal transport parameter validation test passed")

    def test_invalid_inputs(self):
        """Test validation of the source and target point clouds."""
        self.logger.info("Testing optimal transport input validation")

        # Non-2D input.
        x = jnp.array([0.0, 1.0])
        y = jnp.array([[0.0], [1.0]])

        with self.assertRaises(ValueError):
            self.solver(x, y)

        # Different feature dimensions.
        x = jnp.zeros((2, 2))
        y = jnp.zeros((2, 3))

        with self.assertRaises(ValueError):
            self.solver(x, y)

        # Empty source point cloud.
        x = jnp.empty((0, 2))
        y = jnp.zeros((2, 2))

        with self.assertRaises(ValueError):
            self.solver(x, y)

        # Empty target point cloud.
        x = jnp.zeros((2, 2))
        y = jnp.empty((0, 2))

        with self.assertRaises(ValueError):
            self.solver(x, y)

        self.logger.info("✅ Optimal transport input validation test passed")

    def test_small_sinkhorn_problem(self):
        """Test the complete Sinkhorn solver on a very small problem."""
        self.logger.info("Testing optimal transport on a small problem")

        x = jnp.array(
            [
                [0.0],
                [1.0],
            ]
        )

        y = jnp.array(
            [
                [0.0],
                [1.0],
            ]
        )

        output = self.solver(x, y)

        # Verify the shapes of the cost matrix and dual potentials.
        self.assertEqual(
            output.cost_matrix.shape,
            (2, 2),
        )

        self.assertEqual(
            output.fu.shape,
            (2,),
        )

        self.assertEqual(
            output.gv.shape,
            (2,),
        )

        # Verify the iteration limit and convergence diagnostics.
        self.assertLessEqual(
            int(output.num_iterations),
            self.solver.num_iterations,
        )

        self.assertTrue(bool(output.converged))

        self.assertTrue(np.isfinite(np.asarray(output.reg_ot_cost)).all())

        # Verify the expected squared-Euclidean cost matrix.
        np.testing.assert_allclose(
            np.asarray(output.cost_matrix),
            np.array(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ]
            ),
            rtol=1e-12,
            atol=1e-12,
        )

        self.logger.info("✅ Small optimal transport test passed")


def run_tests():
    """Run all optimal transport tests."""

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTransportSolution))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimalTransport))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
