# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Perform empirical Quantile Mapping for bias correction.

The method estimates the empirical distributions of reference and biased
data independently for each feature. Biased values are corrected by mapping
their empirical cumulative probabilities to the corresponding quantiles
of the reference distribution using linear interpolation.
"""

import numpy as np


class QM:
    """
    Basic empirical Quantile Mapping bias corrector.

    Given a reference dataset Y0 and a biased dataset X0, this estimates the
    empirical cumulative distribution function (CDF) of each, feature by
    feature, and corrects biased values by matching their quantiles to those
    of the reference distribution.
    """

    def __init__(self):
        self.n_features = None
        self._sorted_X = []
        self._sorted_Y = []

    @staticmethod
    def _empirical_quantiles(sorted_values):
        """
        Build the probability grid associated with a sorted sample.

        Parameters
        ----------
        sorted_values : numpy.ndarray
            1D array of sample values, already sorted ascending.

        Returns
        -------
        numpy.ndarray
            Cumulative probabilities in (0, 1], one per sample, using the
            plotting-position convention p_k = k / n (k = 1..n).
        """
        # Number of observations in the empirical distribution
        n = sorted_values.shape[0]
        # Example for n = 5:
        #
        #     [1/5, 2/5, 3/5, 4/5, 5/5]
        #
        # which gives:
        #
        #     [0.2, 0.4, 0.6, 0.8, 1.0]
        return np.arange(1, n + 1) / n

    def fit(self, Y0, X0):
        """
        Fit the empirical distributions of the reference and biased data.

        Parameters
        ----------
        Y0 : numpy.ndarray
            Reference dataset with shape (n_samples, n_features), e.g. ERA5.
        X0 : numpy.ndarray
            Biased dataset with shape (n_samples, n_features), e.g. historical model data.
        """
        if Y0.ndim == 1:
            Y0 = Y0.reshape(-1, 1)
        if X0.ndim == 1:
            X0 = X0.reshape(-1, 1)

        # Quantile Mapping is performed feature by feature.
        # Therefore, the reference and biased datasets must contain
        # exactly the same number of features.
        if Y0.shape[1] != X0.shape[1]:
            raise ValueError(
                "Y0 and X0 must have the same number of features: "
                f"{Y0.shape[1]} != {X0.shape[1]}"
            )

        self.n_features = X0.shape[1]
        # Sort the biased values independently for each feature.
        # Sorting gives a simple empirical representation of the
        # biased distribution F_X.
        self._sorted_X = [np.sort(X0[:, i]) for i in range(self.n_features)]

        # Sort the reference values independently for each feature.
        # These values will be used as the empirical inverse CDF
        # (quantile function) F_Y^{-1}.
        self._sorted_Y = [np.sort(Y0[:, i]) for i in range(self.n_features)]

    def predict(self, X0):
        """
        Apply the fitted quantile mapping to new biased data.

        For each feature, a value x is mapped to its empirical
        non-exceedance probability p under the fitted biased distribution,
        then p is mapped back to a value under the fitted reference
        distribution (linear interpolation, clipped to the reference
        sample's range).

        Parameters
        ----------
        X0 : numpy.ndarray
            Data to correct with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Corrected data with shape (n_samples, n_features).
        """
        if self.n_features is None:
            raise RuntimeError("QM.fit() must be called before predict().")

        # Convert a one-dimensional time series into
        # a two-dimensional array with one feature
        if X0.ndim == 1:
            X0 = X0.reshape(-1, 1)

        if X0.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X0.shape[1]}.")

        Z0 = np.zeros_like(X0, dtype=np.float64)

        # Quantile Mapping is performed independently for every feature
        for i in range(self.n_features):
            # Empirical samples fitted for the current feature
            sorted_x = self._sorted_X[i]
            sorted_y = self._sorted_Y[i]

            # p_x describes the probability axis of the biased distribution,
            # p_y describes the probability axis of the reference distribution.
            p_x = self._empirical_quantiles(sorted_x)
            p_y = self._empirical_quantiles(sorted_y)

            # Step 1: map biased values to their empirical probabilities
            # p = F_X(x)
            # np.interp() approximates the empirical CDF by linear
            # interpolation between the fitted biased sample values.
            probabilities = np.interp(
                X0[:, i],
                sorted_x,
                p_x,
                left=p_x[0],
                right=p_x[-1],
            )

            # Step 2: map probabilities to the reference distribution
            # z = F_Y^{-1}(p)
            Z0[:, i] = np.interp(
                probabilities,
                p_y,
                sorted_y,
                left=sorted_y[0],
                right=sorted_y[-1],
            )

        return Z0
