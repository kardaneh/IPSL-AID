# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Bias-correction interfaces and implementations.

This module defines a common interface for bias-correction methods and provides
wrappers for Quantile Mapping and entropy-regularized Optimal Transport.

All correction methods operate on two-dimensional arrays with shape
(n_samples, n_features). In the climate-data workflow, samples correspond
to time steps and features correspond to flattened spatial grid points,
possibly concatenated across several variables.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np

from AID_BC.quantile_mapping import QM
from AID_BC.optimal_transport import OptimalTransport


class BiasCorrector(ABC):
    """
    Common interface for bias-correction methods.

    All arrays follow the convention (n_samples, n_features). In the
    climate-data workflow, samples correspond to time steps and features
    correspond to flattened spatial grid points, possibly concatenated across
    several variables.

    Attributes
    ----------
    is_fitted : bool
        Whether the corrector has been fitted.
    n_features : int or None
        Number of expected input features after fitting.
    """

    method_name = "base"

    def __init__(self):
        self.is_fitted = False
        self.n_features = None

    @abstractmethod
    def fit(
        self,
        reference,
        biased,
    ):
        """
        Fit the bias-correction model.

        Parameters
        ----------
        reference : array-like
            Reference observations with shape (n_reference_samples, n_features),
            for example ERA5 data.
        biased : array-like
            Biased training observations with shape (n_biased_samples, n_features),
            for example historical CMIP6 data.

        Returns
        -------
        BiasCorrector
            Fitted corrector instance.
        """

    @abstractmethod
    def transform(
        self,
        data,
    ):
        """
        Correct new biased data.

        Parameters
        ----------
        data : array-like
            Biased application data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Corrected data with the same shape as data.
        """

    def fit_transform(
        self,
        reference,
        biased,
        data,
    ):
        """
        Fit the corrector and transform application data.

        Parameters
        ----------
        reference : array-like
            Reference observations with shape (n_reference_samples, n_features),
            for example ERA5 data.
        biased : array-like
            Biased training observations with shape (n_biased_samples, n_features).
        data : array-like
            Biased application data with shape (n_samples, n_features),
            for example historical CMIP6 data.

        Returns
        -------
        numpy.ndarray
            Corrected application data.
        """
        self.fit(
            reference=reference,
            biased=biased,
        )

        return self.transform(data)

    @property
    def diagnostics(self):
        """
        Return method diagnostics.

        Returns
        -------
        dict
            Method name, fitting state, and number of input features.
        """
        return {
            "method": self.method_name,
            "is_fitted": self.is_fitted,
            "n_features": self.n_features,
        }

    @staticmethod
    def validate_fit_arrays(
        reference,
        biased,
    ):
        """
        Validate reference and biased training arrays.

        Parameters
        ----------
        reference : numpy.ndarray
            Reference array with shape (n_samples, n_features).
        biased : numpy.ndarray
            Biased training array with shape (n_samples, n_features).

        Raises
        ------
        ValueError
            If an array is not two-dimensional, contains no samples, contains
            non-finite values, or if the feature dimensions differ.
        """
        if reference.ndim != 2:
            raise ValueError(
                "reference must be a 2D array with shape "
                f"(n_samples, n_features), got {reference.shape}"
            )

        if biased.ndim != 2:
            raise ValueError(
                "biased must be a 2D array with shape "
                f"(n_samples, n_features), got {biased.shape}"
            )

        if reference.shape[0] == 0:
            raise ValueError("reference contains no samples.")

        if biased.shape[0] == 0:
            raise ValueError("biased contains no samples.")

        if reference.shape[1] != biased.shape[1]:
            raise ValueError(
                "reference and biased must have the same number "
                f"of features: {reference.shape[1]} != "
                f"{biased.shape[1]}"
            )

        if not np.isfinite(reference).all():
            raise ValueError("reference contains NaN or infinite values.")

        if not np.isfinite(biased).all():
            raise ValueError("biased contains NaN or infinite values.")

    def validate_transform_array(
        self,
        data,
    ):
        """
        Validate application data before transformation.

        Parameters
        ----------
        data : numpy.ndarray
            Application array with shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If the corrector has not been fitted or its feature count is
            unavailable.
        ValueError
            If data is not two-dimensional, has an unexpected number of
            features, or contains non-finite values.
        """
        if not self.is_fitted:
            raise RuntimeError("The bias corrector must be fitted before transform().")

        if data.ndim != 2:
            raise ValueError(
                "data must be a 2D array with shape "
                f"(n_samples, n_features), got {data.shape}"
            )

        if self.n_features is None:
            raise RuntimeError("Internal error: n_features is undefined.")

        if data.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, " f"got {data.shape[1]}."
            )

        if not np.isfinite(data).all():
            raise ValueError("Application data contains NaN or infinite values.")


class QuantileMappingCorrector(BiasCorrector):
    """
    Wrapper around the existing SBCK Quantile Mapping implementation.
    """

    method_name = "qm"

    def __init__(self):
        super().__init__()
        self.model = None

    def fit(
        self,
        reference,
        biased,
    ):
        """
        Fit empirical Quantile Mapping.

        Parameters
        ----------
        reference : array-like
            Reference observations with shape (n_reference_samples, n_features).
        biased : array-like
            Biased training observations with shape (n_biased_samples, n_features).

        Returns
        -------
        QuantileMappingCorrector
            Fitted corrector instance.
        """
        reference = np.asarray(
            reference,
            dtype=np.float32,
        )

        biased = np.asarray(
            biased,
            dtype=np.float32,
        )

        self.validate_fit_arrays(
            reference=reference,
            biased=biased,
        )

        self.n_features = biased.shape[1]

        self.model = QM()

        self.model.fit(
            Y0=reference,
            X0=biased,
        )

        self.is_fitted = True

        return self

    def transform(
        self,
        data,
    ):
        """
        Apply the fitted Quantile Mapping model.

        Parameters
        ----------
        data : array-like
            Biased application data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Corrected data as float32.

        Raises
        ------
        RuntimeError
            If the underlying Quantile Mapping model is unavailable.
        """
        data = np.asarray(
            data,
            dtype=np.float32,
        )

        self.validate_transform_array(data)

        if self.model is None:
            raise RuntimeError("Internal error: QM model is unavailable.")

        corrected = self.model.predict(
            X0=data,
        )

        return np.asarray(
            corrected,
            dtype=np.float32,
        )


class OptimalTransportCorrector(BiasCorrector):
    """
    Optimal Transport bias corrector.

    Parameters
    ----------
    epsilon : float, default=1000
        Entropic regularization strength.
    num_iterations : int, default=15000
        Maximum number of Sinkhorn iterations.
    threshold : float, default=1e-2
        Convergence threshold.
    batch_size : int, default=16
        Number of application samples transported together.
    dtype : {"float32", "float64"}, default="float64"
        Floating-point precision used by NumPy and JAX.
    normalize : bool, default=True
        Whether to fit and apply a shared per-feature normalization.

    Attributes
    ----------
    center : numpy.ndarray or None
        Per-feature normalization center.
    scale : numpy.ndarray or None
        Per-feature normalization scale.
    target_points : jax.Array or None
        Normalized ERA5 target samples.
    target_potential : jax.Array or None
        Target dual potential produced by Sinkhorn.
    solver : AID_BC.optimal_transport.SinkhornSolver or None
        Sinkhorn solver used during fitting.
    transport_function : callable or None
        Vectorized transport function.
    converged : bool or None
        Whether Sinkhorn converged.
    fitted_iterations : int or None
        Number of iterations performed.
    regularized_ot_cost : float or None
        Regularized transport cost.
    """

    method_name = "ot"

    def __init__(
        self,
        epsilon=1000,
        num_iterations=15000,
        threshold=1e-2,
        batch_size=16,
        dtype="float64",
        normalize=True,
    ):
        super().__init__()

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")

        if num_iterations <= 0:
            raise ValueError(
                "num_iterations must be positive, " f"got {num_iterations}."
            )

        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}.")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be either 'float32' or 'float64'.")

        self.epsilon = float(epsilon)
        self.num_iterations = int(num_iterations)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.dtype_name = dtype
        self.normalize = bool(normalize)

        if dtype == "float32":
            self.numpy_dtype = np.float32
            self.jax_dtype = jnp.float32
        else:
            self.numpy_dtype = np.float64
            self.jax_dtype = jnp.float64

        self.center = None
        self.scale = None
        self.target_points = None
        self.target_potential = None
        self.solver = None
        self.transport_function = None
        self.converged = None
        self.fitted_iterations = None
        self.regularized_ot_cost = None

    def _fit_normalization(
        self,
        reference,
        biased,
    ):
        """
        Fit a shared per-feature normalization.

        A shared normalization is fitted from both source and target samples so
        that they remain in the same feature space.

        Parameters
        ----------
        reference : numpy.ndarray
            Reference observations.
        biased : numpy.ndarray
            Biased training observations.

        Returns
        -------
        tuple of numpy.ndarray
            Normalized reference and biased arrays.
        """
        number_reference = reference.shape[0]
        number_biased = biased.shape[0]
        number_total = number_reference + number_biased

        if not self.normalize:
            self.center = np.zeros(
                reference.shape[1],
                dtype=self.numpy_dtype,
            )

            self.scale = np.ones(
                reference.shape[1],
                dtype=self.numpy_dtype,
            )

            return reference, biased

        # Avoid np.concatenate to reduce peak host-memory usage.
        sum_values = reference.sum(axis=0, dtype=np.float64) + biased.sum(
            axis=0, dtype=np.float64
        )

        sum_squared_values = np.square(
            reference,
            dtype=np.float64,
        ).sum(axis=0) + np.square(
            biased,
            dtype=np.float64,
        ).sum(axis=0)

        center = sum_values / number_total

        variance = sum_squared_values / number_total - np.square(center)

        # Protect against small negative values caused by rounding.
        variance = np.maximum(variance, 0.0)

        scale = np.sqrt(variance)

        minimum_scale = 1e-6
        scale = np.maximum(scale, minimum_scale)

        self.center = center.astype(
            self.numpy_dtype,
            copy=False,
        )

        self.scale = scale.astype(
            self.numpy_dtype,
            copy=False,
        )

        return (
            self._normalize(reference),
            self._normalize(biased),
        )

    def _normalize(
        self,
        data,
    ):
        """
        Normalize data with the fitted shared parameters.

        Parameters
        ----------
        data : numpy.ndarray
            Data to normalize.

        Returns
        -------
        numpy.ndarray
            Normalized data in the configured NumPy dtype.

        Raises
        ------
        RuntimeError
            If normalization parameters have not been fitted.
        """
        if self.center is None or self.scale is None:
            raise RuntimeError("Normalization parameters have not been fitted.")

        normalized = (data - self.center) / self.scale

        return normalized.astype(
            self.numpy_dtype,
            copy=False,
        )

    def _denormalize(
        self,
        data,
    ):
        """
        Restore data to the original feature scale.

        Parameters
        ----------
        data : numpy.ndarray
            Normalized data.

        Returns
        -------
        numpy.ndarray
            Data restored to the original scale.

        Raises
        ------
        RuntimeError
            If normalization parameters have not been fitted.
        """
        if self.center is None or self.scale is None:
            raise RuntimeError("Normalization parameters have not been fitted.")

        denormalized = data * self.scale + self.center

        return denormalized.astype(
            self.numpy_dtype,
            copy=False,
        )

    def fit(
        self,
        reference,
        biased,
    ):
        """
        Fit the Optimal Transport map from CMIP6 to ERA5.

        Parameters
        ----------
        reference : array-like
            ERA5 reference observations with shape (n_reference_samples, n_features).
        biased : array-like
            Historical CMIP6 observations with shape (n_biased_samples, n_features).

        Returns
        -------
        OptimalTransportCorrector
            Fitted corrector instance.

        Raises
        ------
        RuntimeError
            If the Sinkhorn solver does not converge.
        """
        reference = np.asarray(
            reference,
            dtype=self.numpy_dtype,
        )

        biased = np.asarray(
            biased,
            dtype=self.numpy_dtype,
        )

        self.validate_fit_arrays(
            reference=reference,
            biased=biased,
        )

        self.n_features = biased.shape[1]

        reference_normalized, biased_normalized = self._fit_normalization(
            reference=reference,
            biased=biased,
        )

        # Source: historical CMIP6.
        source_points = jax.device_put(
            jnp.asarray(
                biased_normalized,
                dtype=self.jax_dtype,
            )
        )

        # Target: ERA5.
        target_points = jax.device_put(
            jnp.asarray(
                reference_normalized,
                dtype=self.jax_dtype,
            )
        )

        self.solver = OptimalTransport(
            epsilon=self.epsilon,
            num_iterations=self.num_iterations,
            threshold=self.threshold,
        )

        output = self.solver(
            source_points,
            target_points,
        )

        # Force execution here rather than during a later operation.
        jax.block_until_ready(output.reg_ot_cost)

        self.converged = bool(np.asarray(output.converged))

        self.fitted_iterations = int(np.asarray(output.num_iterations))

        self.regularized_ot_cost = float(np.asarray(output.reg_ot_cost))

        if not self.converged:
            raise RuntimeError(
                "Sinkhorn did not converge. "
                f"iterations={self.fitted_iterations}, "
                f"maximum={self.num_iterations}, "
                f"epsilon={self.epsilon}. "
                "Try increasing epsilon or num_iterations."
            )

        # For CMIP6 -> ERA5, use the target potential gv and
        # the target points ERA5.
        # self.target_potential = output.gv
        # self.target_points = target_points

        # Store persistent OT data on CPU instead of keeping JAX arrays
        # on the GPU. This avoids GPU-memory accumulation when
        # several correctors (12 month correctors) are kept alive at the same time.
        self.target_potential = np.asarray(output.gv)
        self.target_points = np.asarray(target_points)

        self.transport_function = self.solver.transport_fn(
            potential=self.target_potential,
            y=self.target_points,
        )

        # SinkhornOutput contains the full cost matrix.
        # Release GPU-side temporary arrays after fitting.
        del output
        del source_points
        del target_points
        del biased_normalized
        del reference_normalized

        self.is_fitted = True

        return self

    def transform(
        self,
        data,
    ):
        """
        Apply the fitted Optimal Transport map in batches.

        Parameters
        ----------
        data : array-like
            Biased CMIP6 application data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Corrected data as float32.

        Raises
        ------
        RuntimeError
            If the transport function is unavailable.
        """
        data = np.asarray(
            data,
            dtype=self.numpy_dtype,
        )

        self.validate_transform_array(data)

        if self.transport_function is None:
            raise RuntimeError("Internal error: OT transport function is unavailable.")

        normalized_data = self._normalize(data)

        corrected_batches = []

        number_samples = normalized_data.shape[0]

        for start in range(
            0,
            number_samples,
            self.batch_size,
        ):
            stop = min(
                start + self.batch_size,
                number_samples,
            )

            batch = jax.device_put(
                jnp.asarray(
                    normalized_data[start:stop],
                    dtype=self.jax_dtype,
                )
            )

            corrected_batch = self.transport_function(batch)

            corrected_batch = jax.block_until_ready(corrected_batch)

            corrected_batches.append(
                np.asarray(
                    corrected_batch,
                    dtype=self.numpy_dtype,
                )
            )

            del batch
            del corrected_batch

        corrected_normalized = np.concatenate(
            corrected_batches,
            axis=0,
        )

        corrected = self._denormalize(corrected_normalized)

        return corrected.astype(
            np.float32,
            copy=False,
        )

    @property
    def diagnostics(self):
        """
        Return Optimal Transport fitting diagnostics.

        Returns
        -------
        dict
            Base diagnostics together with backend, devices, solver
            configuration, convergence information, cost, dtype, batch size,
            and normalization state.
        """
        diagnostics = super().diagnostics
        diagnostics.update(
            {
                "backend": jax.default_backend(),
                "devices": [str(device) for device in jax.devices()],
                "epsilon": self.epsilon,
                "threshold": self.threshold,
                "maximum_iterations": self.num_iterations,
                "fitted_iterations": self.fitted_iterations,
                "converged": self.converged,
                "regularized_ot_cost": self.regularized_ot_cost,
                "dtype": self.dtype_name,
                "batch_size": self.batch_size,
                "normalize": self.normalize,
            }
        )
        return diagnostics


def create_bias_corrector(
    method,
    *,
    ot_epsilon=1000,
    ot_num_iterations=15000,
    ot_threshold=1e-2,
    ot_batch_size=16,
    ot_dtype="float64",
    ot_normalize=True,
):
    """
    Create a bias-correction method.

    Parameters
    ----------
    method : {"qm", "ot"}
        Bias-correction method identifier.
    ot_epsilon : float, default=1000
        Entropic regularization used by OT.
    ot_num_iterations : int, default=15000
        Maximum number of Sinkhorn iterations.
    ot_threshold : float, default=1e-2
        Sinkhorn convergence threshold.
    ot_batch_size : int, default=16
        Number of application samples transported together.
    ot_dtype : {"float32", "float64"}, default="float64"
        Floating-point precision used by OT.
    ot_normalize : bool, default=True
        Whether OT applies shared per-feature normalization.

    Returns
    -------
    QuantileMappingCorrector or OptimalTransportCorrector
        New, unfitted corrector instance.

    Raises
    ------
    ValueError
        If method is neither "qm" nor "ot".
    """
    normalized_method = method.strip().lower()

    if normalized_method == "qm":
        return QuantileMappingCorrector()

    if normalized_method == "ot":
        return OptimalTransportCorrector(
            epsilon=ot_epsilon,
            num_iterations=ot_num_iterations,
            threshold=ot_threshold,
            batch_size=ot_batch_size,
            dtype=ot_dtype,
            normalize=ot_normalize,
        )

    raise ValueError(
        f"Unknown bias-correction method: {method}. " "Expected 'qm' or 'ot'."
    )
