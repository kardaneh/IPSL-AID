# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

# ============================================================================
# REFERENCES
# ============================================================================
# Mathematical foundations:
#
# Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of
# Optimal Transport.
#
# Pooladian, A.-A., and Niles-Weed, J. (2021).
# Entropic Estimation of Optimal Transport Maps.
#
# Application to statistical downscaling:
#
# Wan, Z. Y., et al. (2023). Debias Coarsely, Sample Conditionally: Statistical
# Downscaling through Optimal Transport and Probabilistic Diffusion Models.
#
# Sinkhorn implementation reference:
# https://github.com/google-research/swirl-dynamics/blob/main/swirl_dynamics/projects/debiasing/optimal_transport/sinkhorn.py

# ============================================================================
# ACKNOWLEDGMENTS
# ============================================================================
# We thank the swirl_dynamics Authors for making their implementation
# available for further research and development.

"""
Solve entropy-regularized optimal transport using the Sinkhorn algorithm.

The solver computes a squared-Euclidean cost matrix, estimates dual
potentials with Sinkhorn iterations, and constructs transport functions
from the target dual potential.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

Array = jax.Array
jax.config.update("jax_enable_x64", True)


class TransportSolution(NamedTuple):
    """
    Store the output of the entropy-regularized transport solver.

    The source and target dual potentials define the transport plan. The
    remaining fields describe the transport problem and the solver's outcome.
    """

    potentials: tuple[Array, Array]
    cost_matrix: Array
    epsilon: float
    reg_ot_cost: Array
    threshold: float
    converged: Array
    num_iterations: Array

    @property
    def fu(self):
        """Return the source dual potential."""
        return self.potentials[0]

    @property
    def gv(self):
        """Return the target dual potential."""
        return self.potentials[1]

    @property
    def cost(self):
        """Return the pairwise cost matrix."""
        return self.cost_matrix

    @property
    def transport_plan(self):
        """
        Compute the transport plan from the fitted dual potentials.

        Returns
        -------
        jax.Array
            Transport plan with shape (n_source, n_target).
        """
        return jnp.exp(
            (-self.cost_matrix + self.fu[:, None] + self.gv[None, :]) / self.epsilon
        )


def _squared_euclidean_cost(x: Array, y: Array) -> Array:
    """
    Compute pairwise squared-Euclidean distances between two point clouds.

    The dot-product identity avoids constructing an intermediate array with
    shape (n_x, n_y, n_features).

    Parameters
    ----------
    x : jax.Array, shape (n_x, n_features)
        Source samples.
    y : jax.Array, shape (n_y, n_features)
        Target samples.

    Returns
    -------
    jax.Array, shape (n_x, n_y)
        Pairwise squared-Euclidean distances, clipped below at zero to
        remove small negative values caused by floating-point round-off.
    """
    # Compute the squared Euclidean norm of each source and target sample.
    x_norm = jnp.sum(jnp.square(x), axis=1, keepdims=True)  # shape (n_x, 1)
    y_norm = jnp.sum(jnp.square(y), axis=1, keepdims=True).T  # shape (1, n_y)
    dot_products = x @ y.T  # shape (n_x, n_y)
    return jnp.maximum(x_norm + y_norm - 2.0 * dot_products, 0.0)


def _iterate_potentials(cost, epsilon, threshold, num_iterations, eps_marginal):
    """
    Update the source and target dual potentials with Sinkhorn iterations.

    Both empirical distributions are assigned uniform masses. Updates are
    performed in log space, and the iterations stop when the sum of the
    Euclidean norms of successive potential changes reaches the threshold.

    Parameters
    ----------
    cost : jax.Array, shape (n_source, n_target)
        Pairwise squared-Euclidean cost matrix.
    epsilon : float
        Entropic regularization parameter.
    threshold : float
        Convergence threshold for the sum of potential-change norms.
    num_iterations : int
        Maximum number of Sinkhorn iterations.
    eps_marginal : float
        Small value added to each marginal mass before taking its logarithm.

    Returns
    -------
    source_potential : jax.Array, shape (n_source,)
        Fitted source dual potential.
    target_potential : jax.Array, shape (n_target,)
        Fitted target dual potential.
    state : tuple
        Number of iterations and final potential-change error.
    """
    n_source, n_target = cost.shape
    # Represent both point clouds as empirical distributions with uniform masses.
    source_mass = jnp.ones((n_source,)) / n_source
    target_mass = jnp.ones((n_target,)) / n_target
    source_potential = jnp.zeros_like(source_mass)
    target_potential = jnp.zeros_like(target_mass)

    # Precompute terms that do not change during Sinkhorn iterations.
    scaled_cost = -cost / epsilon
    scaled_cost_t = scaled_cost.T
    source_log_mass = jnp.log(source_mass + eps_marginal)
    target_log_mass = jnp.log(target_mass + eps_marginal)

    def update(state):
        """
        Perform one Sinkhorn iteration to update both dual potentials.

        The source potential is updated first using the current target
        potential. The target potential is then updated using the new source
        potential. The convergence error measures the changes in both
        potentials during this iteration.

        Parameters
        ----------
        state : tuple
            Current iteration state containing the iteration count, source
            dual potential, target dual potential, and convergence error.

        Returns
        -------
        tuple
            Updated iteration state containing the incremented iteration
            count, updated source and target dual potentials, and the sum
            of the Euclidean norms of their changes.
        """
        iteration, source, target, _ = state
        previous_source, previous_target = source, target

        # Update the source dual potential using the current target potential.
        row_log_plan = (
            scaled_cost + source[:, None] / epsilon + target[None, :] / epsilon
        )
        row_correction = source_log_mass - logsumexp(row_log_plan, axis=1)
        source = epsilon * row_correction + source

        # Update the target dual potential using the new source potential.
        column_log_plan = (
            scaled_cost_t + target[:, None] / epsilon + source[None, :] / epsilon
        )
        column_correction = target_log_mass - logsumexp(column_log_plan, axis=1)
        target = epsilon * column_correction + target

        # Measure convergence using changes in both dual potentials.
        error = jnp.linalg.norm(previous_source - source) + jnp.linalg.norm(
            previous_target - target
        )
        return iteration + 1, source, target, error

    def continue_iterating(state):
        """
        Check whether Sinkhorn should perform another iteration.

        Iterations continue while the convergence error exceeds the
        configured threshold and the maximum iteration count has not
        been reached.

        Parameters
        ----------
        state : tuple
            Current iteration state containing the iteration count, source
            dual potential, target dual potential, and convergence error.

        Returns
        -------
        jax.Array
            Boolean indicating whether another Sinkhorn iteration is required.
        """
        iteration, _, _, error = state
        return jnp.logical_and(error > threshold, iteration < num_iterations)

    # An infinite initial error guarantees at least one iteration.
    initial_error = jnp.asarray(jnp.inf, dtype=source_potential.dtype)
    count, source_potential, target_potential, final_error = jax.lax.while_loop(
        continue_iterating,
        update,
        (0, source_potential, target_potential, initial_error),
    )
    return source_potential, target_potential, (count, final_error)


class OptimalTransport:
    """
    Solve entropy-regularized optimal transport using Sinkhorn iterations.

    The solver uses the squared-Euclidean cost and uniform empirical
    marginals. The target dual potential can be used to construct a
    transport function for new source samples.

    Parameters
    ----------
    epsilon : float
        Entropic regularization parameter. Must be strictly positive.
    sharding : jax.sharding.Sharding or None, optional
        Optional sharding constraint applied to the cost matrix.
    num_iterations : int, default=100
        Maximum number of Sinkhorn iterations.
    threshold : float, default=1e-3
        Convergence threshold based on changes in the dual potentials.
    eps_marginal : float, default=1e-12
        Small value added to marginal masses before taking their logarithms.
    """

    def __init__(
        self,
        epsilon,
        sharding=None,
        num_iterations=100,
        threshold=1e-3,
        eps_marginal=1e-12,
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be strictly positive, got {epsilon}.")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be strictly positive.")
        if threshold < 0:
            raise ValueError("threshold must be non-negative.")
        if eps_marginal < 0:
            raise ValueError("eps_marginal must be non-negative.")
        self.epsilon = epsilon
        self.sharding = sharding
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.eps_marginal = eps_marginal
        self._compiled_solve = jax.jit(self._solve)

    def __call__(self, x, y) -> TransportSolution:
        """
        Solve the transport problem between source and target samples.

        Parameters
        ----------
        x : jax.Array, shape (n_source, n_features)
            Source samples.
        y : jax.Array, shape (n_target, n_features)
            Target samples.

        Returns
        -------
        TransportSolution
            Fitted dual potentials, cost matrix, transport cost, and
            convergence diagnostics.

        Raises
        ------
        ValueError
            If either point cloud is not two-dimensional, the feature
            dimensions do not match, or either point cloud is empty.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must both be two-dimensional arrays.")
        if x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have the same feature dimension.")
        if x.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Both point clouds must contain at least one sample.")
        return self._compiled_solve(x, y)

    def solve(self, source, target):
        """
        Solve the transport problem using named source and target arguments.

        Parameters
        ----------
        source : jax.Array, shape (n_source, n_features)
            Source samples.
        target : jax.Array, shape (n_target, n_features)
            Target samples.

        Returns
        -------
        TransportSolution
            Fitted transport solution.
        """
        return self(source, target)

    def _solve(self, x, y):
        """
        Compute the transport solution and its convergence diagnostics.

        Parameters
        ----------
        x : jax.Array, shape (n_source, n_features)
            Source samples.
        y : jax.Array, shape (n_target, n_features)
            Target samples.

        Returns
        -------
        TransportSolution
            Dual potentials, cost matrix, transport cost, and convergence state.
        """
        # Compute the squared-Euclidean distance between each source
        # sample and each target sample.
        cost = _squared_euclidean_cost(x, y)
        if self.sharding is not None:
            cost = jax.lax.with_sharding_constraint(cost, self.sharding)

        # Run Sinkhorn iterations to estimate the source and target
        # dual potentials. Also retrieve the number of iterations
        # performed and the final convergence error.
        source, target, (count, error) = _iterate_potentials(
            cost,
            self.epsilon,
            self.threshold,
            self.num_iterations,
            self.eps_marginal,
        )

        # Scale the transport costs by the entropic regularization
        # parameter before reconstructing the transport plan.
        scaled_cost = -cost / self.epsilon

        # Combine the scaled costs with the source and target dual
        # potentials to obtain the logarithm of the transport plan.
        log_plan = (
            scaled_cost
            + source[:, None] / self.epsilon
            + target[None, :] / self.epsilon
        )

        # Recover the transport plan by exponentiating its logarithm.
        # The plan has shape (n_source, n_target).
        plan = jnp.exp(log_plan)

        # Compute the transport cost as the sum of the elementwise
        # product of the transport plan and the cost matrix.
        former_cost = jnp.sum(plan * cost, axis=(-2, -1))

        # Return the fitted potentials, transport problem parameters,
        # transport cost, and convergence diagnostics.
        # Convergence is reached when the final change in the dual
        # potentials is less than or equal to the configured threshold.
        return TransportSolution(
            potentials=(source, target),
            cost_matrix=cost,
            epsilon=self.epsilon,
            reg_ot_cost=former_cost,
            threshold=self.threshold,
            converged=error <= self.threshold,
            num_iterations=count,
        )

    def _potential_at(self, point, potential, y, weights=None):
        """
        Evaluate the entropic source potential at one source sample.

        The potential combines the target dual potential, squared-Euclidean
        costs, and target marginal weights using a weighted log-sum-exp.

        Parameters
        ----------
        point : jax.Array, shape (n_features,)
            Source sample at which to evaluate the potential.
        potential : jax.Array, shape (n_target,)
            Target dual potential obtained from Sinkhorn iterations.
        y : jax.Array, shape (n_target, n_features)
            Target samples associated with the dual potential.
        weights : jax.Array or None
            Target marginal weights. Uniform weights are used when None.

        Returns
        -------
        jax.Array
            Scalar value of the entropic source potential.

        Raises
        ------
        ValueError
            If the source and target samples have different feature dimensions.
        """
        # Convert the source sample into a two-dimensional array so that
        # its shape is compatible with the pairwise cost computation.
        point = jnp.atleast_2d(point)

        # Assign uniform weights to the target samples when no
        # marginal weights are provided.
        if weights is None:
            weights = jnp.ones((y.shape[0],)) / y.shape[0]

        # The source sample and target samples must have the same
        # number of features to compute squared-Euclidean distances.
        if point.shape[-1] != y.shape[-1]:
            raise ValueError("point and y must share the feature dimension.")

        # Compute the squared-Euclidean distance between the source
        # sample and each target sample.
        cost = jnp.squeeze(_squared_euclidean_cost(point, y))

        # Combine the target dual potential and the transport costs,
        # then scale the result by the entropic regularization parameter.
        scaled_scores = (potential - cost) / self.epsilon

        # Evaluate the entropic source potential using a weighted
        # log-sum-exp for numerical stability.
        # The result is a scalar because the potential is evaluated
        # at a single source sample.
        return jnp.squeeze(
            -self.epsilon
            * logsumexp(
                scaled_scores,
                b=weights,
                axis=-1,
            )
        )

    def transport_fn(self, potential, y, weights=None):
        """Build a vectorized transport function from the target dual potential.

        The transport map is computed from the gradient of the entropic source
        potential: T(x) = x - 0.5 * grad(f_epsilon(x)). The factor 0.5 comes
        from using squared-Euclidean costs without a factor of one half.

        Parameters
        ----------
        potential : jax.Array, shape (n_target,)
            Target dual potential obtained from Sinkhorn iterations.
        y : jax.Array, shape (n_target, n_features)
            Target samples associated with the dual potential.
        weights : jax.Array or None, optional
            Target marginal weights. Uniform weights are used when None.

        Returns
        -------
        callable
            Function accepting an array of shape (n_samples, n_features)
            and returning the transported samples with the same shape.
        """

        # Differentiate the entropic potential for each source sample.
        def single_point_map(point):
            """
            Transport one source sample using the gradient of the entropic potential.

            Parameters
            ----------
            point : jax.Array, shape (n_features,)
                Source sample to transport.

            Returns
            -------
            jax.Array, shape (n_features,)
                Transported source sample.
            """

            def potential_value(x):
                """
                Evaluate the entropic potential at one source sample.

                Parameters
                ----------
                x : jax.Array, shape (n_features,)
                    Source sample at which to evaluate the potential.

                Returns
                -------
                jax.Array
                    Scalar value of the entropic potential.
                """
                return self._potential_at(x, potential, y, weights)

            return point - 0.5 * jax.grad(potential_value)(point)

        return jax.vmap(single_point_map, in_axes=0, out_axes=0)

    def transport_fn_direct(self, potential, y, weights=None):
        """
        Build a direct transport function using the target dual potential.

        This implementation evaluates the transport weights using exponentials
        without log-sum-exp stabilization and may be numerically unstable.

        Parameters
        ----------
        potential : jax.Array, shape (n_target,)
            Target dual potential obtained from Sinkhorn iterations.
        y : jax.Array, shape (n_target, n_features)
            Target samples associated with the dual potential.
        weights : jax.Array or None, optional
            Target marginal weights. Uniform weights are used when None.

        Returns
        -------
        callable
            Function transporting one source sample of shape (n_features,)
            to a target-space sample with the same shape.

        Raises
        ------
        ValueError
            If the target dual potential is not one-dimensional or its length
            does not match the number of target samples.
        """
        if potential.ndim != 1:
            raise ValueError("potential must be one-dimensional.")
        if potential.shape[0] != y.shape[0]:
            raise ValueError("potential and y must have the same sample count.")
        if weights is None:
            weights = jnp.ones((y.shape[0],)) / y.shape[0]

        def transport_one(point):
            """
            Transport one source sample using the direct entropic transport map.

            The transport weights are computed from the target dual potential and
            the squared-Euclidean distances to the target samples. The transported
            sample is obtained as a weighted average of the target samples.

            Parameters
            ----------
            point : jax.Array, shape (n_features,)
                Source sample to transport.

            Returns
            -------
            jax.Array, shape (n_features,)
                Transported source sample.
            """
            # Compute squared-Euclidean distances to the target samples.
            cost = jnp.squeeze(_squared_euclidean_cost(point, y))
            # Compute the transport weights from the target dual potential.
            unnormalized = jnp.exp((potential - cost) / self.epsilon) * weights
            # Compute the weighted average of the target samples.
            return jnp.sum(y * unnormalized[:, None], axis=0) / jnp.sum(unnormalized)

        return transport_one
