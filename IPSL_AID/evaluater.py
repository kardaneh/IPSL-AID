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
import pandas as pd
from tqdm import tqdm
from IPSL_AID.diagnostics import (
    plot_validation_hexbin,
    plot_comparison_hexbin,
    plot_validation_pdfs,
    plot_power_spectra,
    plot_qq_quantiles,
    plot_surface,
    plot_zoom_comparison,
    plot_MAE_map,
    plot_error_map,
    plot_metrics_heatmap,
    plot_validation_mvcorr,
    plot_validation_mvcorr_space,
    plot_temporal_series_comparison,
)


class MetricTracker:
    """
    A utility class for tracking and computing statistics of metric values.

    This class maintains a running average of metric values and provides
    methods to compute mean and root mean squared values.

    Attributes
    ----------
    value : float
        Cumulative weighted sum of metric values
    count : int
        Total number of samples processed

    Examples
    --------
    >>> tracker = MetricTracker()
    >>> tracker.update(10.0, 5)  # value=10.0, count=5 samples
    >>> tracker.update(20.0, 3)  # value=20.0, count=3 samples
    >>> print(tracker.getmean())  # (10*5 + 20*3) / (5+3) = 110/8 = 13.75
    13.75
    >>> print(tracker.getsqrtmean())  # sqrt(13.75)
    3.7080992435478315
    """

    def __init__(self):
        """
        Initialize MetricTracker with zero values.
        """
        self.reset()

    def reset(self):
        """
        Reset all tracked values to zero.

        Returns
        -------
        None
        """
        self.value = 0.0
        self.count = 0
        self.value_sq = 0.0

    def update(self, value, count):
        """
        Update the tracker with new metric values.

        Parameters
        ----------
        value : float
            The metric value to add
        count : int
            Number of samples this value represents (weight)

        Returns
        -------
        None
        """
        self.count += count
        self.value += value * count
        self.value_sq += (value**2) * count

    def getmean(self):
        """
        Calculate the mean of all tracked values.

        Returns
        -------
        float
            Weighted mean of all values: total_value / total_count

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute mean with zero samples")
        return self.value / self.count

    def getstd(self):
        """
        Calculate the standard deviation of all tracked values.

        Returns
        -------
        float
            Weighted standard deviation of all values:
            sqrt(E(x^2) - (E(x))^2)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute std with zero samples")
        mean = self.getmean()
        variance = self.value_sq / self.count - mean**2
        return np.sqrt(max(variance, 0.0))  # numerical safety

    def getsqrtmean(self):
        """
        Calculate the square root of the mean of all tracked values.

        Returns
        -------
        float
            Square root of the weighted mean: sqrt(total_value / total_count)

        Raises
        ------
        ZeroDivisionError
            If no values have been added (count == 0)
        """
        return np.sqrt(self.getmean())


def mae_all(pred, true):
    """
    Calculate Mean Absolute Error (MAE) between predicted and true values.

    Computes the MAE metric and returns both the number of elements and
    the mean absolute error value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values

    Returns
    -------
    tuple
        (num_elements, mae_value) where:
        - num_elements (int): Total number of elements in the tensors
        - mae_value (torch.Tensor): Mean absolute error value

    Examples
    --------
    >>> pred = torch.tensor([1.0, 2.0, 3.0])
    >>> true = torch.tensor([1.1, 1.9, 3.2])
    >>> num_elements, mae = mae_all(pred, true)
    >>> print(f"MAE: {mae.item():.4f}, Elements: {num_elements}")
    MAE: 0.1333, Elements: 3

    Notes
    -----
    The MAE is calculated as: mean(abs(pred - true))
    This function is useful for tracking metrics with MetricTracker
    """
    num_elements = pred.numel()
    mae_value = torch.mean(torch.abs(pred - true))
    return num_elements, mae_value


def nmae_all(pred, true, eps=1e-8):
    """
    Normalized Mean Absolute Error (NMAE).
    NMAE = MAE(pred, true) / mean(abs(true))

    Computes the NMAE metric and returns both the number of elements and
    the normalized mean absolute error value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values
    eps : float
        Small value to avoid division by zero

    Returns
    -------
    tuple
        (num_elements, mae_value) where:
        - num_elements (int): Total number of elements in the tensors
        - mae_value (torch.Tensor): Mean absolute error value

    Examples
    --------
    >>> pred = torch.tensor([1.0, 2.0, 3.0])
    >>> true = torch.tensor([1.1, 1.9, 3.2])
    >>> num_elements, nmae = nmae_all(pred, true)
    >>> print(f"NMAE: {nmae.item():.4f}, Elements: {num_elements}")
    NMAE: 0.047059, Elements: 3

    Notes
    -----
    The NMAE is calculated as: MAE(pred, true) / mean(abs(true))
    This function is useful for tracking metrics with MetricTracker
    """
    num_elements = pred.numel()
    mae = torch.mean(torch.abs(pred - true))
    norm = torch.mean(torch.abs(true)) + eps
    nmae = mae / norm
    return num_elements, nmae


# To verify with Kazem
def crps_ensemble_all(pred_ens, true):
    """
    Continuous Ranked Probability Score (CRPS) for an ensemble.

    Computes the CRPS metric for ensemble predictions and returns both
    the number of elements and the mean CRPS value.

    Parameters
    ----------
    pred_ens : torch.Tensor
        Ensemble predictions, shape [N_ens, N_pixels]
    true : torch.Tensor
        Ground truth values, shape [N_pixels]

    Returns
    -------
    tuple
        (num_elements, crps_mean) where:
        - num_elements (int): Total number of elements in the tensors
        - crps_mean (torch.Tensor): Mean CRPS

    Notes
    -----
    The CRPS for an ensemble is computed as:

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X and X' are independent ensemble members and y is the
    observation.
    """
    # Number of ensemble members
    n = pred_ens.shape[0]

    # Sort ensemble
    pred_ens_sorted, _ = torch.sort(pred_ens, dim=0)

    # Term 1: E|X - y|
    term1 = torch.mean(torch.abs(pred_ens - true.unsqueeze(0)), dim=0)

    # Term 2: ensemble spread term
    diff = pred_ens_sorted[1:] - pred_ens_sorted[:-1]
    weight = torch.arange(1, n, device=pred_ens.device) * torch.arange(
        n - 1, 0, -1, device=pred_ens.device
    )
    term2 = torch.sum(diff * weight.unsqueeze(1), dim=0) / (n**2)

    crps_pixel = term1 - term2  # [N_pixels]

    # Final aggregation
    num_elements = crps_pixel.numel()
    crps_mean = crps_pixel.mean()

    return num_elements, crps_mean


def rmse_all(pred, true):
    """
    Calculate Root Mean Square Error (RMSE) between predicted and true values.

    Computes the RMSE metric and returns both the number of elements and
    the root mean square error value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values

    Returns
    -------
    tuple
        (num_elements, rmse_value) where:
        - num_elements (int): Total number of elements in the tensors
        - rmse_value (torch.Tensor): Root mean square error value

    Examples
    --------
    >>> pred = torch.tensor([1.0, 2.0, 3.0])
    >>> true = torch.tensor([1.1, 1.9, 3.2])
    >>> num_elements, rmse = rmse_all(pred, true)
    >>> print(f"RMSE: {rmse.item():.4f}, Elements: {num_elements}")
    RMSE: 0.1414, Elements: 3

    Notes
    -----
    The RMSE is calculated as: sqrt(mean((pred - true)^2))
    This function is useful for tracking metrics with MetricTracker
    """
    num_elements = pred.numel()
    mse = torch.mean((pred - true) ** 2)
    rmse_value = torch.sqrt(mse)
    return num_elements, rmse_value


def r2_all(pred, true):
    """
    Calculate R2 (coefficient of determination) between predicted and true values.

    Computes the R2 metric and returns both the number of elements and
    the R2 value.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model
    true : torch.Tensor
        Ground truth values

    Returns
    -------
    tuple
        (num_elements, r2_value) where:
        - num_elements (int): Total number of elements in the tensors
        - r2_value (torch.Tensor): R2 score

    Notes
    -----
    R2 is calculated as:

        R2 = 1 - sum((true - pred)^2) / sum((true - mean(true))^2)

    This implementation is fully torch-based and works on CPU and GPU.
    """

    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")

    eps = 1e-12  # Small value to avoid division by zero when variance is zero
    num_elements = pred.numel()

    # Flatten
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Residual sum of squares
    ss_res = torch.sum((true_flat - pred_flat) ** 2)

    # Total sum of squares
    true_mean = torch.mean(true_flat)
    ss_tot = torch.sum((true_flat - true_mean) ** 2)

    # R2 score
    r2_value = 1.0 - ss_res / (ss_tot + eps)

    return num_elements, r2_value


def pearson_all(pred, true):
    """
    Compute the Pearson correlation coefficient between predicted and
    ground truth values using torch.corrcoef.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model.
    true : torch.Tensor
        Ground truth values.

    Returns
    -------
    tuple
        (num_elements, pearson_value) where:
        - num_elements (int): Total number of elements in the tensors.
        - pearson_value (torch.Tensor): Pearson correlation coefficient.

    Notes
    -----
    The Pearson correlation coefficient is defined as:

        rho = Cov(pred, true) / (std(pred) * std(true))
    """

    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: {pred.shape} vs {true.shape}")

    num_elements = pred.numel()

    # Flatten tensors to 1D vectors
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Stack into a 2 x N matrix required by torch.corrcoef
    stacked = torch.stack([pred_flat, true_flat], dim=0)

    # Compute correlation matrix
    corr_matrix = torch.corrcoef(stacked)

    # Extract Pearson correlation coefficient between
    # predictions (row 0) and truth (row 1)
    pearson_value = corr_matrix[0, 1]

    return num_elements, pearson_value


def kl_divergence_all(pred, true):
    """
    Compute the Kullback–Leibler (KL) divergence between predicted and
    ground truth distributions using histogram-based estimation.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values from the model.
    true : torch.Tensor
        Ground truth values.

    Returns
    -------
    tuple
        (num_elements, kl_value) where:
        - num_elements (int): Total number of elements in the tensors.
        - kl_value (torch.Tensor): KL divergence value.

    Notes
    -----
    The KL divergence is defined as:

        KL(P|Q) = sum_i P_i * log(P_i / Q_i)

    where:
        - P represents the true distribution
        - Q represents the predicted distribution
    """

    if pred.shape != true.shape:
        raise RuntimeError(f"Shape mismatch: {pred.shape} vs {true.shape}")

    num_elements = pred.numel()

    n_bins = 100
    eps = 1e-12

    # Flatten tensors to 1D vectors
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)

    # Combine for percentile computation
    all_values = torch.cat([pred_flat, true_flat])

    # Percentile clipping
    data_min = torch.quantile(all_values, 0.0025)
    data_max = torch.quantile(all_values, 0.995)

    data_range = data_max - data_min

    x_min = data_min - 0.05 * data_range
    x_max = data_max + 0.05 * data_range

    hist_pred = torch.histc(pred_flat, bins=n_bins, min=x_min.item(), max=x_max.item())
    hist_true = torch.histc(true_flat, bins=n_bins, min=x_min.item(), max=x_max.item())

    # Add epsilon
    hist_pred = hist_pred + eps
    hist_true = hist_true + eps

    # Normalize to probability mass
    hist_pred = hist_pred / hist_pred.sum()
    hist_true = hist_true / hist_true.sum()

    # KL divergence
    kl_value = torch.sum(hist_true * torch.log(hist_true / hist_pred))

    return num_elements, kl_value


def denormalize(
    data,
    stats,
    norm_type,
    device,
    var_name=None,
    data_type=None,
    debug=False,
    logger=None,
):
    """
    Denormalize a data tensor using the inverse of the normalization operation.

    Parameters
    ----------
    data : torch.Tensor
        Normalized tensor to denormalize.
    stats : object
        Object containing the required statistics.
    norm_type : str
        Normalization type used originally.
    device : torch.device
        Device for tensor operations.
    var_name : str, optional
        Variable name for debugging.
    data_type : str, optional
        Data type for debugging (e.g., "residual", "coarse").
    debug : bool, optional
        Enable debug logging.
    logger : Logger, optional
        Logger instance for debug output.
    """
    # Add debug logging at the start
    if debug and logger:
        # Create context string
        context = ""
        if var_name:
            context = f" for {var_name}"
        if data_type:
            context += f" ({data_type})"

        logger.info(
            f"Denormalizing{context} with type '{norm_type}'\n"
            f" └── Denormalization stats;\n"
            f"   └── vmin: {getattr(stats, 'vmin', None)}\n"
            f"   └── vmax: {getattr(stats, 'vmax', None)}\n"
            f"   └── vmean: {getattr(stats, 'vmean', None)}\n"
            f"   └── vstd: {getattr(stats, 'vstd', None)}\n"
            f"   └── median: {getattr(stats, 'median', None)}\n"
            f"   └── iqr: {getattr(stats, 'iqr', None)}\n"
            f"   └── q1: {getattr(stats, 'q1', None)}\n"
            f"   └── q3: {getattr(stats, 'q3', None)}"
        )
    # ------------------ MIN-MAX ------------------
    if norm_type == "minmax":
        vmin = torch.tensor(stats.vmin, dtype=data.dtype, device=device)
        vmax = torch.tensor(stats.vmax, dtype=data.dtype, device=device)
        denom = vmax - vmin
        if denom == 0:
            return torch.zeros_like(data)
        return data * denom + vmin

    # ------------------ MIN-MAX [-1, 1] -----------------
    elif norm_type == "minmax_11":
        vmin = torch.tensor(stats.vmin, dtype=data.dtype, device=device)
        vmax = torch.tensor(stats.vmax, dtype=data.dtype, device=device)
        denom = vmax - vmin
        if denom == 0:
            return torch.zeros_like(data)
        return ((data + 1) / 2) * denom + vmin

    # ------------------ STANDARD -----------------
    elif norm_type == "standard":
        mean = torch.tensor(stats.vmean, dtype=data.dtype, device=device)
        std = torch.tensor(stats.vstd, dtype=data.dtype, device=device)
        if std == 0:
            return torch.zeros_like(data)
        return data * std + mean

    # ------------------ ROBUST -------------------
    elif norm_type == "robust":
        median = torch.tensor(stats.median, dtype=data.dtype, device=device)
        iqr = torch.tensor(stats.iqr, dtype=data.dtype, device=device)
        if iqr == 0:
            return torch.zeros_like(data)
        return data * iqr + median

    # ------------------ LOG1P + MIN-MAX ------------------
    elif norm_type == "log1p_minmax":
        log_min = torch.tensor(stats.vmin, dtype=data.dtype, device=device)
        log_max = torch.tensor(stats.vmax, dtype=data.dtype, device=device)
        denom = log_max - log_min

        if denom == 0:
            return torch.zeros_like(data)

        log_data = data * denom + log_min
        return torch.expm1(log_data)

    # ------------------ LOG1P + STANDARD ------------------
    elif norm_type == "log1p_standard":
        mean = torch.tensor(stats.vmean, dtype=data.dtype, device=device)
        std = torch.tensor(stats.vstd, dtype=data.dtype, device=device)

        if std == 0:
            return torch.zeros_like(data)

        log_data = data * std + mean
        return torch.expm1(log_data)

    else:
        raise ValueError(f"Unsupported norm_type '{norm_type}'")


@torch.no_grad()
def edm_sampler(
    model,
    image_input,
    class_labels=None,
    num_steps=40,
    sigma_min=0.02,
    sigma_max=80.0,
    rho=7,
    S_churn=40,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    """
    EDM sampler for diffusion model inference.
    Original work: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
    Original source: https://github.com/NVlabs/edm

    Parameters
    ----------
    model : torch.nn.Module
        Diffusion model
    image_input : torch.Tensor
        Conditioning input (coarse + constants)
    class_labels : torch.Tensor, optional
        Time conditioning labels
    num_steps : int, optional
        Number of sampling steps
    sigma_min : float, optional
        Minimum noise level
    sigma_max : float, optional
        Maximum noise level
    rho : float, optional
        Time step exponent
    S_churn : int, optional
        Stochasticity parameter
    S_min : float, optional
        Minimum stochasticity threshold
    S_max : float, optional
        Maximum stochasticity threshold
    S_noise : float, optional
        Noise scale for stochasticity

    Returns
    -------
    torch.Tensor
        Generated residual predictions
    """
    batch_size, _, H, W = image_input.shape

    # Get the actual model (unwrap DataParallel if needed)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # init noise
    init_noise = torch.randn(
        (batch_size, model.out_channels, H, W),
        dtype=image_input.dtype,
        device=image_input.device,
    )

    # Adjust noise levels based on what's supported by the model.
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(
        num_steps, dtype=image_input.dtype, device=image_input.device
    )
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = init_noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = model.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = model(x_hat, t_hat, image_input, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(x_next, t_next, image_input, class_labels).to(
                torch.float64
            )
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next.detach()


@torch.no_grad()
def sampler(
    epoch,
    batch_idx,
    model,
    image_input,
    class_labels=None,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="heun",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    S_churn=40,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    logger=None,
):
    """
    General sampler for diffusion model inference with multiple configurations.
    Original work: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.
    Original source: https://github.com/NVlabs/edm

    Parameters
    ----------
    model : torch.nn.Module
        Diffusion model
    image_input : torch.Tensor
        Conditioning input (coarse + constants)
    class_labels : torch.Tensor, optional
        Time conditioning labels
    num_steps : int, optional
        Number of sampling steps
    sigma_min : float, optional
        Minimum noise level
    sigma_max : float, optional
        Maximum noise level
    rho : float, optional
        Time step exponent for EDM discretization
    solver : str, optional
        Solver type: 'euler' or 'heun'
    discretization : str, optional
        Discretization type: 'vp', 've', 'iddpm', or 'edm'
    schedule : str, optional
        Noise schedule: 'vp', 've', or 'linear'
    scaling : str, optional
        Scaling type: 'vp' or 'none'
    epsilon_s : float, optional
        Small epsilon for VP schedule
    C_1 : float, optional
        Constant for IDDPM discretization
    C_2 : float, optional
        Constant for IDDPM discretization
    M : int, optional
        Number of steps for IDDPM discretization
    alpha : float, optional
        Parameter for Heun's method
    S_churn : int, optional
        Stochasticity parameter
    S_min : float, optional
        Minimum stochasticity threshold
    S_max : float, optional
        Maximum stochasticity threshold
    S_noise : float, optional
        Noise scale for stochasticity
    logger : logging.Logger, optional
        Logger instance for logging sampler parameters

    Returns
    -------
    torch.Tensor
        Generated residual predictions
    """
    # Only the original asserts with messages
    assert solver in [
        "euler",
        "heun",
    ], f"Solver must be 'euler' or 'heun', but got '{solver}'"
    assert (
        discretization in ["vp", "ve", "iddpm", "edm"]
    ), f"Discretization must be 'vp', 've', 'iddpm' or 'edm', but got '{discretization}'"
    assert schedule in [
        "vp",
        "ve",
        "linear",
    ], f"Schedule must be 'vp', 've' or 'linear', but got '{schedule}'"
    assert scaling in [
        "vp",
        "none",
    ], f"Scaling must be 'vp' or 'none', but got '{scaling}'"

    batch_size, _, H, W = image_input.shape

    # Get the actual model (unwrap DataParallel if needed)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Initialize noise
    latents = torch.randn(
        (batch_size, model.out_channels, H, W),
        dtype=image_input.dtype,
        device=image_input.device,
    )

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: (
        lambda t: (np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1) ** 0.5
    )
    vp_sigma_deriv = lambda beta_d, beta_min: (
        lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = lambda beta_d, beta_min: (
        lambda sigma: (
            ((beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min)
            / beta_d
        )
    )
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Log sampler parameters if logger is provided
    if logger is not None and epoch == 0 and batch_idx == 0:
        logger.info("=== Sampler Parameters ===")
        logger.info(f" └── num_steps: {num_steps}")
        logger.info(f" └── solver: {solver}")
        logger.info(f" └── discretization: {discretization}")
        logger.info(f" └── schedule: {schedule}")
        logger.info(f" └── scaling: {scaling}")
        logger.info(f" └── sigma_min: {sigma_min}")
        logger.info(f" └── sigma_max: {sigma_max}")
        logger.info(f" └── rho: {rho}")
        logger.info(f" └── S_churn: {S_churn}")
        logger.info(f" └── S_min: {S_min}")
        logger.info(f" └── S_max: {S_max}")
        logger.info(f" └── S_noise: {S_noise}")
        logger.info(f" └── epsilon_s: {epsilon_s}")
        logger.info(f" └── C_1: {C_1}")
        logger.info(f" └── C_2: {C_2}")
        logger.info(f" └── M: {M}")
        logger.info(f" └── alpha: {alpha}")
        logger.info("==========================")

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, model.sigma_min)
    sigma_max = min(sigma_max, model.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(
        num_steps, dtype=image_input.dtype, device=image_input.device
    )
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=image_input.dtype, device=image_input.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=image_input.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        assert discretization == "edm"
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == "linear"
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == "none"
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(model.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(image_input.dtype) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= sigma(t_cur) <= S_max
            else 0
        )
        t_hat = sigma_inv(model.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * S_noise * torch.randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = model(x_hat / s(t_hat), sigma(t_hat), image_input, class_labels).to(
            image_input.dtype
        )
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == "heun"
            denoised = model(
                x_prime / s(t_prime), sigma(t_prime), image_input, class_labels
            ).to(image_input.dtype)
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next.detach()


def reconstruct_original_layout(
    epoch, args, paths, steps, all_data, dataset, device, logger
):
    """
    Robust reconstruction using dataset information directly.

    Parameters:
    -----------
    all_data : dict
        Dictionary containing lists of batches for:
        - 'predictions': model predictions [B, C, H, W]
        - 'coarse': coarse resolution data [B, C, H, W]
        - 'fine': fine resolution ground truth [B, C, H, W]
        - 'lat': latitude coordinates [B, H]
        - 'lon': longitude coordinates [B, W]
    dataset : torch.utils.data.Dataset
        The validation dataset instance
    device : torch.device
        Device to store tensors on
    logger : Logger
        Logger instance for logging

    Returns:
    --------
    dict: Reconstructed data with metadata
    """
    # Get dataset parameters
    time_batchs = len(dataset.time_batchs)
    sbatch = dataset.sbatch
    total_dataset_samples = len(dataset)  # time_batchs * sbatch

    # dataset_times = dataset.loaded_dfs.time.values

    # Get total samples from all batches
    total_batch_samples = sum(batch.shape[0] for batch in all_data["predictions"])

    logger.info("Dataset reconstruction info:")
    logger.info(f" └── time_batchs: {time_batchs}")
    logger.info(f" └── sbatch: {sbatch}")
    logger.info(f" └── total dataset samples: {total_dataset_samples}")
    logger.info(f" └── total batch samples: {total_batch_samples}")

    # Handle different scenarios
    if total_batch_samples > total_dataset_samples:
        error_msg = (
            f"More batch samples ({total_batch_samples}) than dataset samples ({total_dataset_samples})! "
            f"Something is wrong with the DataLoader."
        )
        logger.error(error_msg)
        raise
    elif total_batch_samples < total_dataset_samples:
        logger.info(
            f"Note: Batch samples ({total_batch_samples}) < dataset samples ({total_dataset_samples})"
        )
        logger.info("This is normal if DataLoader has drop_last=True")

    # Get sample shape
    pred_shape = all_data["predictions"][0].shape[1:]  # [C, H, W]
    C, H, W = pred_shape

    logger.info(f"Sample shape: C={C}, H={H}, W={W}")

    # Initialize reconstruction arrays
    reconstructions = {}
    for key in ["predictions", "coarse", "fine"]:
        reconstructions[key] = torch.zeros(
            time_batchs, sbatch, C, H, W, device=device, dtype=all_data[key][0].dtype
        )
        logger.info(f"Initialized {key} with shape: {reconstructions[key].shape}")

    reconstructions["lat"] = torch.zeros(
        time_batchs, sbatch, H, device=device, dtype=all_data["lat"][0].dtype
    )
    reconstructions["lon"] = torch.zeros(
        time_batchs, sbatch, W, device=device, dtype=all_data["lon"][0].dtype
    )
    logger.info(f"Initialized lat with shape: {reconstructions['lat'].shape}")
    logger.info(f"Initialized lon with shape: {reconstructions['lon'].shape}")

    # Create position tracking
    position_filled = torch.zeros(time_batchs, sbatch, dtype=torch.bool, device=device)

    # Map each dataset index to position
    index_to_position = {}
    for idx in range(total_dataset_samples):
        sindex = idx % sbatch
        tindex = idx // sbatch
        index_to_position[idx] = (tindex, sindex)

    logger.info(f"Created index mapping for {total_dataset_samples} samples")

    # Reconstruct using dataset indices
    dataset_idx = 0
    total_reconstructed = 0

    logger.info("Starting reconstruction process...")

    for batch_idx in range(len(all_data["predictions"])):
        batch = all_data["predictions"][batch_idx]
        batch_size = batch.shape[0]

        logger.info(
            f"Processing batch {batch_idx+1}/{len(all_data['predictions'])} with size {batch_size}"
        )

        for i_in_batch in range(batch_size):
            # We can only reconstruct up to dataset samples
            if dataset_idx >= total_dataset_samples:
                logger.warning(
                    f"Stopping at dataset_idx {dataset_idx} (dataset has {total_dataset_samples} samples)"
                )
                break

            tindex, sindex = index_to_position[dataset_idx]

            # Store all data
            for key in ["predictions", "coarse", "fine"]:
                reconstructions[key][tindex, sindex] = all_data[key][batch_idx][
                    i_in_batch
                ]

            reconstructions["lat"][tindex, sindex] = all_data["lat"][batch_idx][
                i_in_batch
            ]
            reconstructions["lon"][tindex, sindex] = all_data["lon"][batch_idx][
                i_in_batch
            ]

            position_filled[tindex, sindex] = True
            total_reconstructed += 1
            dataset_idx += 1

        # Free memory for this batch
        for key in ("predictions", "coarse", "fine", "lat", "lon"):
            all_data[key][batch_idx] = None

        # Break if we've reached dataset limit
        if dataset_idx >= total_dataset_samples:
            break

    logger.info(f"Successfully reconstructed {total_reconstructed} samples")

    # Check results
    filled_count = position_filled.sum().item()
    if filled_count != total_reconstructed:
        logger.warning(
            f"filled_count ({filled_count}) != total_reconstructed ({total_reconstructed})"
        )

    if filled_count < total_dataset_samples:
        missing = total_dataset_samples - filled_count
        logger.info(
            f"Note: {missing}/{total_dataset_samples} samples not reconstructed"
        )
        logger.info("This is expected with drop_last=True in DataLoader")

    # Metadata
    metadata = {
        "time_batchs": time_batchs,
        "sbatch": sbatch,
        "total_dataset_samples": total_dataset_samples,
        "total_batch_samples": total_batch_samples,
        "total_reconstructed": total_reconstructed,
        "position_filled": position_filled,
        "index_to_position": index_to_position,
        "filled_ratio": filled_count / total_dataset_samples
        if total_dataset_samples > 0
        else 0,
        "reconstruction_device": str(device),
    }

    logger.info("Reconstruction completed successfully")

    # Check if we need to combine spatial blocks for inference
    if args.run_type in ["inference", "inference_regional"]:
        logger.info(
            "Inference mode is active - combining spatial blocks to reconstruct full domain..."
        )

        # Get evaluation slices directly from the DataPreprocessor
        if hasattr(dataset, "eval_slices"):
            eval_slices = dataset.eval_slices
            logger.info(f"Found {len(eval_slices)} evaluation slices")

            # Determine the spatial extent covered by evaluation slices.
            # In regional inference, slices may not start at index 0, so the domain size
            # is computed from the min/max slice indices.
            lat_min = min(s[0] for s in eval_slices)
            lat_max = max(s[1] for s in eval_slices)

            lon_min = min(s[2] for s in eval_slices)
            lon_max = max(s[3] for s in eval_slices)

            covered_H = lat_max - lat_min
            covered_W = lon_max - lon_min

            logger.info(f"Dataset dimensions: H={dataset.H}, W={dataset.W}")
            logger.info(f"Blocks cover: H={covered_H}, W={covered_W}")

            # Initialize coordinate arrays
            lat_reconstructed = torch.zeros(covered_H, device=device)
            lon_reconstructed = torch.zeros(covered_W, device=device)

            # Track which coordinates we've filled (must fill all!)
            lat_filled = torch.zeros(covered_H, dtype=torch.bool, device=device)
            lon_filled = torch.zeros(covered_W, dtype=torch.bool, device=device)

            # Initialize arrays for the COVERED area
            combined_data = {}
            for key in ["predictions", "coarse", "fine"]:
                combined_data[key] = torch.zeros(
                    time_batchs,
                    C,
                    covered_H,
                    covered_W,
                    device=device,
                    dtype=reconstructions[key].dtype,
                )

            # Track grid coverage (must cover all!)
            coverage_mask = torch.zeros(
                covered_H, covered_W, dtype=torch.bool, device=device
            )

            # Combine blocks and reconstruct coordinates
            blocks_placed = 0
            for t in range(time_batchs):
                for spatial_idx, (lat_start, lat_end, lon_start, lon_end) in enumerate(
                    eval_slices
                ):
                    # Shift slice indices into the local reconstruction coordinate system.
                    # This is required for regional inference where slices do not start at 0.
                    # For global inference lat_min=lon_min=0 so indices remain unchanged.
                    lat_start -= lat_min
                    lat_end -= lat_min
                    lon_start -= lon_min
                    lon_end -= lon_min

                    if spatial_idx >= sbatch:
                        error_msg = (
                            f"CRITICAL ERROR: Slice index {spatial_idx} exceeds sbatch {sbatch}. "
                            f"eval_slices has {len(eval_slices)} slices but only {sbatch} spatial blocks reconstructed."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Place block in combined array
                    for key in ["predictions", "coarse", "fine"]:
                        combined_data[key][
                            t, :, lat_start:lat_end, lon_start:lon_end
                        ] = reconstructions[key][t, spatial_idx]

                    # Reconstruct LATITUDE coordinates from this block
                    block_lat = reconstructions["lat"][t, spatial_idx]  # [H_block]
                    lat_reconstructed[lat_start:lat_end] = block_lat
                    lat_filled[lat_start:lat_end] = True

                    # Reconstruct LONGITUDE coordinates from this block
                    block_lon = reconstructions["lon"][t, spatial_idx]  # [W_block]
                    lon_reconstructed[lon_start:lon_end] = block_lon
                    lon_filled[lon_start:lon_end] = True

                    # Mark grid coverage
                    coverage_mask[lat_start:lat_end, lon_start:lon_end] = True
                    blocks_placed += 1

            logger.info(f"Combined {blocks_placed} spatial blocks")

            # VERIFY COMPLETE COVERAGE - RAISE ERROR IF INCOMPLETE

            # Check latitude coordinate coverage
            lat_missing = (~lat_filled).sum().item()
            if lat_missing > 0:
                missing_indices = torch.nonzero(~lat_filled).squeeze().cpu().numpy()
                error_msg = (
                    f"CRITICAL ERROR: Latitude coordinate reconstruction incomplete!\n"
                    f"Missing {lat_missing}/{covered_H} latitude coordinates.\n"
                    f"Missing indices: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}\n"
                    f"This indicates blocks don't cover the full latitude range."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check longitude coordinate coverage
            lon_missing = (~lon_filled).sum().item()
            if lon_missing > 0:
                missing_indices = torch.nonzero(~lon_filled).squeeze().cpu().numpy()
                error_msg = (
                    f"CRITICAL ERROR: Longitude coordinate reconstruction incomplete!\n"
                    f"Missing {lon_missing}/{covered_W} longitude coordinates.\n"
                    f"Missing indices: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}\n"
                    f"This indicates blocks don't cover the full longitude range."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Check grid coverage
            uncovered_cells = (~coverage_mask).sum().item()
            if uncovered_cells > 0:
                # Find where coverage is missing
                missing_mask = ~coverage_mask
                missing_positions = torch.nonzero(missing_mask)

                error_msg = (
                    f"CRITICAL ERROR: Grid coverage incomplete!\n"
                    f"Missing {uncovered_cells}/{covered_H*covered_W} grid cells.\n"
                    f"Coverage: {coverage_mask.sum().item()/(covered_H*covered_W)*100:.1f}%\n"
                    f"First 10 missing positions (lat, lon): {missing_positions[:10].cpu().numpy().tolist()}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Fix longitude discontinuity when blocks cross the 0°/360° meridian.
            # np.unwrap keeps the longitude coordinate monotonic
            # Ex: 358,359,0,1 to 358,359,360,361
            lon_reconstructed = torch.from_numpy(
                np.rad2deg(np.unwrap(np.deg2rad(lon_reconstructed.cpu().numpy())))
            ).to(device)

            # All checks passed - reconstruction is complete
            logger.info("✅ Coordinate reconstruction complete")
            logger.info("✅ Grid coverage complete")
            logger.info(
                f"Latitude range: {lat_reconstructed.min():.2f} to {lat_reconstructed.max():.2f}"
            )
            logger.info(
                f"Longitude range: {lon_reconstructed.min():.2f} to {lon_reconstructed.max():.2f}"
            )

            # Add reconstruction info to metadata
            metadata["coverage_info"] = {
                "covered_H": covered_H,
                "covered_W": covered_W,
                "full_H": dataset.H,
                "full_W": dataset.W,
                "coverage_complete": True,
                "coordinates_complete": True,
                "lat_range": [
                    lat_reconstructed.min().item(),
                    lat_reconstructed.max().item(),
                ],
                "lon_range": [
                    lon_reconstructed.min().item(),
                    lon_reconstructed.max().item(),
                ],
                "lat_reconstructed": lat_reconstructed.cpu(),
                "lon_reconstructed": lon_reconstructed.cpu(),
            }

            # Store reconstructed coordinates in reconstructions dict
            reconstructions["lat_reconstructed"] = lat_reconstructed
            reconstructions["lon_reconstructed"] = lon_reconstructed

            # Add combined data to reconstructions dict
            reconstructions["combined"] = combined_data

        else:
            logger.error(
                "Could not find eval_slices in dataset. Cannot combine spatial blocks."
            )
            raise AttributeError(
                "Dataset missing 'eval_slices' attribute for inference reconstruction."
            )

    logger.info(f"Generating block wise plots for epoch {epoch}...")

    # Loop through spatial blocks
    for spatial_idx in range(sbatch):
        # Extract data for this spatial block
        # shape: [time_batchs, C, H, W]
        predictions_block = reconstructions["predictions"][:, spatial_idx]
        fine_block = reconstructions["fine"][:, spatial_idx]
        coarse_block = reconstructions["coarse"][:, spatial_idx]
        # lat_block = reconstructions['lat'][:, spatial_idx]
        # lon_block = reconstructions['lon'][:, spatial_idx]

        # 0. QQ Plot
        save_path = plot_qq_quantiles(
            predictions_block,  # [time_batchs, C, H, W]
            fine_block,  # [time_batchs, C, H, W]
            coarse_block,  # [time_batchs, C, H, W]
            variable_names=args.varnames_list,
            units=None,  # You might want to add units to args
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            filename=f"{args.run_type}_qq_epoch_{epoch}_spatial_block_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved QQ plot to {save_path}")

        # 1. Validation Hexbin Plot
        save_path = plot_validation_hexbin(
            predictions=predictions_block,
            targets=fine_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_validation_hexbin_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved validation hexbin plot to: {save_path}")

        # 2. Comparison Hexbin Plot
        save_path = plot_comparison_hexbin(
            predictions=predictions_block,
            targets=fine_block,
            coarse_inputs=coarse_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_comparison_hexbin_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved comparison hexbin plot to: {save_path}")

        # 3. Validation PDFs Plot
        save_path = plot_validation_pdfs(
            predictions=predictions_block,
            targets=fine_block,
            coarse_inputs=coarse_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_validation_pdfs_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved validation PDFs plot to: {save_path}")

        # 4. Power Spectra Plot
        dlon = getattr(steps, "d_longitude", None)
        dlat = getattr(steps, "d_latitude", None)
        assert dlon is not None, "d_longitude not found in steps"
        assert dlat is not None, "d_latitude not found in steps"

        save_path = plot_power_spectra(
            predictions=predictions_block,
            targets=fine_block,
            coarse_inputs=coarse_block,
            dlat=dlat,
            dlon=dlon,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_power_spectra_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved power spectra plot to: {save_path}")

        # 5. MAE map plot (time-averaged)

        # Latitude and longitude coordinates for this spatial block.
        # Coordinates are time-invariant, so we take them from the first time index (t = 0).
        first_time_idx = 0
        # Get coordinates for this spatial block
        lat_block = reconstructions["lat"][first_time_idx, spatial_idx]  # [H]
        lon_block = reconstructions["lon"][first_time_idx, spatial_idx]  # [W]

        save_path = plot_MAE_map(
            predictions=predictions_block,  # [T, C, H, W]
            targets=fine_block,  # [T, C, H, W]
            lat_1d=lat_block,  # [H]
            lon_1d=lon_block,  # [W]
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_mae_map_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved MAE map to: {save_path}")

        # 6. Multivariate Correlation Maps
        # Convert 1D lat/lon to 2D meshgrid
        lat_2d, lon_2d = torch.meshgrid(lat_block, lon_block, indexing="ij")

        save_path = plot_validation_mvcorr(
            predictions=predictions_block,  # [T, C, H, W]
            targets=fine_block,  # [T, C, H, W]
            coarse_inputs=coarse_block,  # optional
            lat=lat_2d.numpy(),
            lon=lon_2d.numpy(),
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_mvcorr_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved multivariate correlation map to: {save_path}")

        # 7. Surface plot
        coarse = reconstructions["coarse"][
            first_time_idx : first_time_idx + 1, spatial_idx
        ]
        fine = reconstructions["fine"][first_time_idx : first_time_idx + 1, spatial_idx]
        pred = reconstructions["predictions"][
            first_time_idx : first_time_idx + 1, spatial_idx
        ]

        save_path = plot_surface(
            predictions=pred,
            targets=fine,
            coarse_inputs=coarse,
            lat_1d=lat_block,
            lon_1d=lon_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_plot_surface_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved surface plot to: {save_path}")

        # 8. Temporal series
        save_path = plot_temporal_series_comparison(
            predictions=predictions_block,
            targets=fine_block,
            # coarse_inputs=coarse_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_temporal_series_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved temporal series plot to: {save_path}")

        # 9. Multivariate spatial correlation time series
        save_path = plot_validation_mvcorr_space(
            predictions=predictions_block,
            targets=fine_block,
            coarse_inputs=coarse_block,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_mvcorr_space_epoch_{epoch}_sblock_{spatial_idx:03d}.png",
            save_dir=paths.results,
        )

        logger.info(
            f"Saved multivariate spatial correlation time series to: {save_path}"
        )

    # For inference mode, also generate full domain plots
    if args.run_type in ["inference", "inference_regional"]:
        assert (
            "combined" in reconstructions
        ), "Combined data not found in reconstructions for inference mode"
        logger.info(
            f"Generating full domain plots for inference mode, epoch {epoch}..."
        )

        # Get combined data for full domain
        predictions_full = reconstructions["combined"][
            "predictions"
        ]  # [time_batchs, C, covered_H, covered_W]
        fine_full = reconstructions["combined"][
            "fine"
        ]  # [time_batchs, C, covered_H, covered_W]
        coarse_full = reconstructions["combined"][
            "coarse"
        ]  # [time_batchs, C, covered_H, covered_W]
        lat_full = reconstructions["lat_reconstructed"]  # [covered_H]
        lon_full = reconstructions["lon_reconstructed"]  # [covered_W]

        # Generate full domain versions of all plots
        # 0. QQ Plot for full domain (averaged over space)
        save_path = plot_qq_quantiles(
            predictions_full,  # [time_batchs, C, H, W]
            fine_full,  # [time_batchs, C, H, W]
            coarse_full,  # [time_batchs, C, H, W]
            variable_names=args.varnames_list,
            units=None,
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            filename=f"{args.run_type}_full_domain_qq_epoch_{epoch}.png",
            save_dir=paths.results,
            save_npz=True,
        )
        logger.info(f"Saved full domain QQ plot to {save_path}")

        # 1. Validation Hexbin Plot for full domain
        save_path = plot_validation_hexbin(
            predictions=predictions_full,
            targets=fine_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_validation_hexbin_epoch_{epoch}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved full domain validation hexbin plot to: {save_path}")

        # 2. Comparison Hexbin Plot for full domain
        save_path = plot_comparison_hexbin(
            predictions=predictions_full,
            targets=fine_full,
            coarse_inputs=coarse_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_comparison_hexbin_epoch_{epoch}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved full domain comparison hexbin plot to: {save_path}")

        # 3. Validation PDFs Plot for full domain
        save_path = plot_validation_pdfs(
            predictions=predictions_full,
            targets=fine_full,
            coarse_inputs=coarse_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_validation_pdfs_epoch_{epoch}.png",
            save_dir=paths.results,
            save_npz=True,
        )
        logger.info(f"Saved full domain validation PDFs plot to: {save_path}")

        # 4. Power Spectra Plot for full domain
        dlon = getattr(steps, "d_longitude", None)
        dlat = getattr(steps, "d_latitude", None)
        assert dlon is not None, "d_longitude not found in steps"
        assert dlat is not None, "d_latitude not found in steps"

        save_path = plot_power_spectra(
            predictions=predictions_full,
            targets=fine_full,
            coarse_inputs=coarse_full,
            dlat=dlat,
            dlon=dlon,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_power_spectra_epoch_{epoch}.png",
            save_dir=paths.results,
            save_npz=True,
        )
        logger.info(f"Saved full domain power spectra plot to: {save_path}")

        # 5. MAE map Plot for full domain
        save_path = plot_MAE_map(
            predictions=predictions_full,
            targets=fine_full,
            lat_1d=lat_full,
            lon_1d=lon_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_mae_map_epoch_{epoch}.png",
            save_dir=paths.results,
        )
        logger.info(f"Saved full domain MAE map to: {save_path}")

        # 6. Error map Plot for full domain
        save_path = plot_error_map(
            predictions=predictions_full,
            targets=fine_full,
            lat_1d=lat_full,
            lon_1d=lon_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_error_map_epoch_{epoch}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved full domain error map to: {save_path}")

        # 7. Surface plots for first few time steps of full domain
        num_time_steps_to_plot = min(3, time_batchs)
        for time_idx in range(num_time_steps_to_plot):
            # Extract single time step
            pred_single_time = predictions_full[time_idx : time_idx + 1]  # [1, C, H, W]
            fine_single_time = fine_full[time_idx : time_idx + 1]  # [1, C, H, W]
            coarse_single_time = coarse_full[time_idx : time_idx + 1]  # [1, C, H, W]

            tindex = dataset.time_batchs[time_idx]

            timestamp = pd.to_datetime(
                dataset.loaded_dfs.time.values[tindex]
            ).to_pydatetime()

            save_path = plot_surface(
                predictions=pred_single_time,
                targets=fine_single_time,
                coarse_inputs=coarse_single_time,
                lat_1d=lat_full,
                lon_1d=lon_full,
                timestamp=timestamp,
                variable_names=args.varnames_list,
                filename=f"{args.run_type}_full_domain_surface_epoch_{epoch}_time_{time_idx:03d}.png",
                save_dir=paths.results,
            )
            logger.info(
                f"Saved full domain surface plot (time {time_idx}) to: {save_path}"
            )

            # Zoom comparison plot for full domain (only for global inference)
            if args.run_type == "inference":
                save_path = plot_zoom_comparison(
                    predictions=pred_single_time,
                    targets=fine_single_time,
                    lat_1d=lat_full,
                    lon_1d=lon_full,
                    variable_names=args.varnames_list,
                    filename=f"{args.run_type}_full_domain_zoom_comparison_epoch_{epoch}_time_{time_idx:03d}.png",
                    save_dir=paths.results,
                )
                logger.info(
                    f"Saved full domain zoom comparison (time {time_idx}) to: {save_path}"
                )

        # 8. Multivariate Correlation Maps for full domain
        # Convert 1D lat/lon to 2D meshgrid
        lat_2d_full, lon_2d_full = torch.meshgrid(lat_full, lon_full, indexing="ij")

        save_path = plot_validation_mvcorr(
            predictions=predictions_full,  # [T, C, H, W]
            targets=fine_full,  # [T, C, H, W]
            coarse_inputs=coarse_full,
            lat=lat_2d_full.numpy(),
            lon=lon_2d_full.numpy(),
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_mvcorr_epoch_{epoch}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved full domain multivariate correlation map to: {save_path}")

        # 9. Temporal series for full domain
        save_path = plot_temporal_series_comparison(
            predictions=predictions_full,
            targets=fine_full,
            # coarse_inputs=coarse_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_temporal_series_epoch_{epoch}.png",
            save_dir=paths.results,
        )

        logger.info(f"Saved full domain temporal series plot to: {save_path}")

        # 10. Multivariate spatial correlation time series for full domain
        save_path = plot_validation_mvcorr_space(
            predictions=predictions_full,
            targets=fine_full,
            coarse_inputs=coarse_full,
            variable_names=args.varnames_list,
            filename=f"{args.run_type}_full_domain_mvcorr_space_epoch_{epoch}.png",
            save_dir=paths.results,
        )

        logger.info(
            f"Saved full domain multivariate spatial correlation time series to: {save_path}"
        )

    return {"data": reconstructions, "metadata": metadata, "device": device}


def generate_residuals_norm(
    model,
    features,
    labels,
    targets,
    loss_fn,
    args,
    device,
    logger,
    epoch=0,
    batch_idx=0,
    inference_type="sampler",
):
    """
    Generate normalized residuals for all variables.

    Parameters
    ----------
    model : torch.nn.Module
        Diffusion model
    features : torch.Tensor
        Input feature tensor provided to the model
    labels : torch.Tensor
        Conditioning labels provided to the model
    targets : torch.Tensor
        Ground truth target tensor used for noise injection in direct inference
    loss_fn : callable
        Loss function
    args : argparse.Namespace
        Command line arguments
    device : torch.device
        Training device
    logger : Logger
        Logger instance
    epoch : int
        Current epoch number
    inference_type : str, optional
        Inference mode, either "direct" (deterministic) or "sampler"
        (stochastic diffusion sampling)

    Returns
    -------
    torch.Tensor
        [B, C, H, W] residuals in normalized space
    """
    # Generate samples for metrics calculation
    # Choose direct for rapid evaluation, sampler for full quality
    if inference_type == "direct":
        if args.debug:
            logger.info("Using direct inference/evaluation mode (deterministic)")

        if args.precond == "unet":
            # Direct prediction for unet
            generated_residuals = model(features, class_labels=labels)
        else:
            rnd_normal = torch.randn([targets.shape[0], 1, 1, 1], device=targets.device)
            sigma = (rnd_normal * loss_fn.P_std + loss_fn.P_mean).exp()
            noisy_targets = targets + torch.randn_like(targets) * sigma
            generated_residuals = model(noisy_targets, sigma, features, labels)

    elif inference_type == "sampler":
        if args.precond == "unet":
            raise ValueError("UNet does not support sampler inference")
        if args.debug and logger is not None:
            logger.info("Using sampler inference/evaluation mode (stochastic)")
            logger.info(f"Starting EDM sampler with {args.num_steps} steps")
        generated_residuals = sampler(
            epoch,
            batch_idx,
            model,
            features,
            labels,
            num_steps=args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            solver=args.solver,
            S_churn=args.s_churn,
            S_min=args.s_min,
            S_max=args.s_max,
            S_noise=args.s_noise,
            logger=logger,
        )
    else:
        logger.error(f"Unknown inference_type: {inference_type}")
        raise

    return generated_residuals


def run_validation(
    model,
    valid_dataset,
    valid_loader,
    loss_fn,
    norm_mapping,
    normalization_type,
    index_mapping,
    args,
    steps,
    device,
    logger,
    epoch,
    writer=None,
    plot_every_n_epochs=None,
    paths=None,
    compute_crps=False,
    crps_batch_size=2,
    crps_ensemble_size=10,
):
    """
    Run validation on the model.

    Parameters
    ----------
    model : torch.nn.Module
        Diffusion model
    valid_loader : DataLoader
        Validation data loader
    loss_fn : callable
        Loss function
    norm_mapping : dict
        Normalization statistics
    normalization_type : EasyDict
        Normalization types for each variable
    args : argparse.Namespace
        Command line arguments
    device : torch.device
        Training device
    logger : Logger
        Logger instance
    epoch : int
        Current epoch number
    writer : SummaryWriter, optional
        TensorBoard writer
    plot_every_n_epochs : int, optional
        Frequency (in epochs) at which validation plots are generated
    paths : dict, optional
        Paths used for saving reconstructions and plots
    compute_crps : bool, optional
        Whether to compute CRPS using stochastic ensemble sampling
    crps_batch_size : int, optional
        Number of validation batches used for CRPS computation
    crps_ensemble_size : int, optional
        Number of ensemble members used to estimate CRPS

    Returns
    -------
    tuple
        (avg_val_loss, val_metrics) - average validation loss and metrics dictionary
    """
    # Define available metrics
    metric_names = [
        "MAE",
        "NMAE",
        "RMSE",
        "R2",
        "PEARSON",
        "KL",
    ]  # You can add more metrics here like ["MAE", "MSE", "RMSE"]
    metric_funcs = {
        "MAE": mae_all,
        "NMAE": nmae_all,
        "RMSE": rmse_all,
        "R2": r2_all,
        "PEARSON": pearson_all,
        "KL": kl_divergence_all,
        # You can add more metrics here:
        # "MSE": mse_all,
    }

    # Add CRPS only if requested
    if compute_crps:
        metric_names.append("CRPS")
        metric_funcs["CRPS"] = crps_ensemble_all

    # Separate deterministic metrics from CRPS.
    # CRPS is handled separately due to its stochastic and expensive nature.
    deterministic_metrics = [m for m in metric_names if m != "CRPS"]

    model.eval()
    val_loss = MetricTracker()

    # Create metrics for both model predictions and coarse baseline.
    # This is done in two steps because deterministic metrics (MAE, NMAE)
    # are computed for both model predictions and the coarse baseline,
    # whereas CRPS is a probabilistic metric and is only defined for
    # stochastic model outputs (no coarse vs fine CRPS).
    val_metrics = {}
    for k in args.varnames_list:
        for m in deterministic_metrics:
            val_metrics[f"{k}_pred_vs_fine_{m}"] = (
                MetricTracker()
            )  # Model prediction vs true fine
            val_metrics[f"{k}_coarse_vs_fine_{m}"] = (
                MetricTracker()
            )  # Coarse vs true fine (baseline)
        if compute_crps:
            val_metrics[f"{k}_pred_vs_fine_CRPS"] = MetricTracker()

    # Add average metrics across all variables for each metric type
    for m in deterministic_metrics:
        val_metrics[f"average_pred_vs_fine_{m}"] = MetricTracker()
        val_metrics[f"average_coarse_vs_fine_{m}"] = MetricTracker()
    if compute_crps:
        val_metrics["average_pred_vs_fine_CRPS"] = MetricTracker()

    all_data = {"predictions": [], "coarse": [], "fine": [], "lat": [], "lon": []}

    crps_batches = []

    logger.info(f"Running validation for epoch {epoch}...")
    logger.info(f"EDM Sampler parameters: steps={args.num_steps}")

    with torch.no_grad():
        val_loop = tqdm(
            enumerate(valid_loader),
            total=len(valid_loader),
            desc=f"Validation Epoch {epoch}",
        )

        for batch_idx, batch in val_loop:
            # Move data to device
            features = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            coarse = batch["coarse"].to(device)
            # coarse_norm = batch["coarse_norm"].to(device)
            # Number of variables (channels)
            n_vars = len(args.varnames_list)
            # Extract normalized coarse field from model inputs
            coarse_norm = features[:, :n_vars]
            fine = batch["fine"].to(device)
            lat_batch = batch["corrdinates"]["lat"].to(device)
            lon_batch = batch["corrdinates"]["lon"].to(device)

            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"Validation batch idx:{batch_idx}\n"
                    f"features shape:{features.shape}, targets shape:{targets.shape}\n"
                    f"coarse shape:{coarse.shape}, fine shape:{fine.shape}\n"
                    f"lat shape:{lat_batch.shape}, lon shape:{lon_batch.shape}"
                )

            # Prepare labels
            if args.time_normalization == "linear":
                labels = torch.stack(
                    (batch["doy"].to(device), batch["hour"].to(device)), dim=1
                )
            elif args.time_normalization == "cos_sin":
                labels = torch.stack(
                    (
                        batch["doy_sin"].to(device),
                        batch["doy_cos"].to(device),
                        batch["hour_sin"].to(device),
                        batch["hour_cos"].to(device),
                    ),
                    dim=1,
                )

            # Calculate validation loss
            with torch.amp.autocast(device_type=device.type, dtype=features.dtype):
                loss = loss_fn(model, targets, features, labels)
                # unet loss is a scalar, so no need for mean
                if args.precond != "unet":
                    loss = loss.mean()

            val_loss.update(loss.item(), targets.shape[0])

            # Store a limited number of batches for CRPS computation.
            # CRPS is expensive, so we only keep the first crps_batch_size batches
            # and reuse the existing features and labels.
            if compute_crps and len(crps_batches) < crps_batch_size:
                crps_batches.append(
                    {
                        "features": features,
                        "labels": labels,
                        "batch": batch,
                    }
                )

            # Track batch-level averages for overall metrics for each metric type
            batch_metric_sums = {
                m: {"pred": MetricTracker(), "coarse": MetricTracker()}
                for m in deterministic_metrics
            }

            generated_residual = generate_residuals_norm(
                model=model,
                features=features,
                labels=labels,
                targets=targets,
                loss_fn=loss_fn,
                args=args,
                device=device,
                logger=logger,
                epoch=epoch,
                batch_idx=batch_idx,
                inference_type=args.inference_type,
            )

            batch_predictions = []

            # Reconstruct final images
            for var_name in args.varnames_list:
                # Get the correct channel index for this variable
                iv = index_mapping[var_name]

                # Reconstruct final image: coarse + residual
                coarse_var_norm = coarse_norm[:, iv : iv + 1]
                final_prediction_norm = (
                    coarse_var_norm + generated_residual[:, iv : iv + 1]
                )

                # Calculate metrics against ground truth fine data
                fine_var = fine[:, iv : iv + 1]
                coarse_var = coarse[:, iv : iv + 1]

                norm_type = normalization_type[var_name]
                if norm_type.startswith("log1p"):
                    stats_fine = norm_mapping[f"{var_name}_fine_log"]
                else:
                    stats_fine = norm_mapping[f"{var_name}_fine"]

                final_prediction = denormalize(
                    final_prediction_norm,
                    stats_fine,
                    norm_type,
                    device,
                    var_name=var_name,
                    data_type="fine",
                    debug=args.debug,
                    logger=logger,
                )

                batch_predictions.append(final_prediction)

                # Calculate all metrics for this variable
                for metric_name in deterministic_metrics:
                    metric_func = metric_funcs[metric_name]

                    # Model prediction vs fine
                    num_elements_pred, metric_value_pred = metric_func(
                        final_prediction, fine_var
                    )
                    val_metrics[f"{var_name}_pred_vs_fine_{metric_name}"].update(
                        metric_value_pred.item(), num_elements_pred
                    )

                    # Coarse vs fine (baseline metric)
                    num_elements_coarse, metric_value_coarse = metric_func(
                        coarse_var, fine_var
                    )
                    val_metrics[f"{var_name}_coarse_vs_fine_{metric_name}"].update(
                        metric_value_coarse.item(), num_elements_coarse
                    )

                    # Accumulate for batch averages
                    batch_metric_sums[metric_name]["pred"].update(
                        metric_value_pred.item(), num_elements_pred
                    )
                    batch_metric_sums[metric_name]["coarse"].update(
                        metric_value_coarse.item(), num_elements_coarse
                    )

            final_prediction_batch = torch.cat(batch_predictions, dim=1)  # [B, C, H, W]

            # Store only needed data for reconstruction
            # Validation outputs are accumulated and immediately moved to CPU
            # to avoid CUDA out-of-memory errors.
            all_data["predictions"].append(final_prediction_batch.detach().cpu())
            all_data["coarse"].append(coarse.detach().cpu())
            all_data["fine"].append(fine.detach().cpu())
            all_data["lat"].append(lat_batch.detach().cpu())  # [B, H]
            all_data["lon"].append(lon_batch.detach().cpu())  # [B, W]

            # Update overall average metrics for this batch for each metric type
            for metric_name in deterministic_metrics:
                batch_avg_pred = batch_metric_sums[metric_name]["pred"].getmean()
                batch_avg_coarse = batch_metric_sums[metric_name]["coarse"].getmean()
                val_metrics[f"average_pred_vs_fine_{metric_name}"].update(
                    batch_avg_pred, 1
                )
                val_metrics[f"average_coarse_vs_fine_{metric_name}"].update(
                    batch_avg_coarse, 1
                )

            # Update progress bar (show first metric by default)
            primary_metric = deterministic_metrics[0]
            batch_avg_pred = batch_metric_sums[primary_metric]["pred"].getmean()
            batch_avg_coarse = batch_metric_sums[primary_metric]["coarse"].getmean()

            val_loop.set_postfix(
                {
                    "Val Loss": f"{loss.item():.4f}",
                    "Avg Val Loss": f"{val_loss.getmean():.4f}",
                    f"Avg Pred {primary_metric}": f"{batch_avg_pred:.4f}",
                    f"Avg Coarse {primary_metric}": f"{batch_avg_coarse:.4f}",
                }
            )

    torch.cuda.empty_cache()
    avg_val_loss = val_loss.getmean()

    # To verify with Kazem
    # Compute CRPS only if requested and if some batches were collected.
    # CRPS is evaluated using an ensemble of stochastic sampler runs.

    if compute_crps and len(crps_batches) > 0:
        logger.info(
            "CRPS configuration summary:\n"
            f" └── Number of CRPS batches: {len(crps_batches)}\n"
            f" └── Ensemble size: {crps_ensemble_size}"
        )

        for item in tqdm(crps_batches, desc="CRPS batches", total=len(crps_batches)):
            features = item["features"]
            labels = item["labels"]
            batch = item["batch"]

            # Generate an ensemble of predictions using the sampler
            ens_preds = []

            for _ in tqdm(range(crps_ensemble_size), desc="CRPS ensemble", leave=False):
                generated_residual = generate_residuals_norm(
                    model=model,
                    features=features,
                    labels=labels,
                    targets=batch["targets"].to(device),
                    loss_fn=loss_fn,
                    args=args,
                    device=device,
                    logger=None,
                    epoch=epoch,
                    batch_idx=-1,  # not tied to validation loop
                    inference_type="sampler",
                )

                # Reconstruct final prediction
                reconstructed_vars = []

                # Extract normalized coarse field from inputs
                n_vars = len(args.varnames_list)
                coarse_norm = features[:, :n_vars]

                for var_name in args.varnames_list:
                    iv = index_mapping[var_name]

                    # coarse_var_norm = batch["coarse_norm"][:, iv:iv+1].to(device)
                    coarse_var_norm = coarse_norm[:, iv : iv + 1]

                    final_pred_norm = (
                        coarse_var_norm + generated_residual[:, iv : iv + 1]
                    )

                    norm_type = normalization_type[var_name]
                    if norm_type.startswith("log1p"):
                        stats_fine = norm_mapping[f"{var_name}_fine_log"]
                    else:
                        stats_fine = norm_mapping[f"{var_name}_fine"]

                    final_pred = denormalize(
                        final_pred_norm,
                        stats_fine,
                        norm_type,
                        device,
                        var_name=var_name,
                        data_type="fine",
                        debug=args.debug,
                        logger=logger,
                    )

                    reconstructed_vars.append(final_pred)

                # Final reconstructed prediction for this ensemble member
                final_prediction = torch.cat(reconstructed_vars, dim=1)  # [B, C, H, W]
                ens_preds.append(final_prediction)

            # Compute CRPS per variable
            pred_ens = torch.stack(ens_preds, dim=0)  # [N_ens, B, C, H, W]

            for var_name in args.varnames_list:
                iv = index_mapping[var_name]

                pred_ens_var = pred_ens[:, :, iv : iv + 1, :, :]  # [N_ens, B, 1, H, W]
                fine_var = batch["fine"][:, iv : iv + 1].to(device)

                pred_ens_flat = pred_ens_var.reshape(crps_ensemble_size, -1)
                true_flat = fine_var.reshape(-1)

                # Compute CRPS per variable using ensemble predictions.
                num_elem, crps_mean = crps_ensemble_all(pred_ens_flat, true_flat)

                # Update per-variable CRPS tracker
                val_metrics[f"{var_name}_pred_vs_fine_CRPS"].update(
                    crps_mean.item(), num_elem
                )

                # Global average CRPS tracker
                val_metrics["average_pred_vs_fine_CRPS"].update(
                    crps_mean.item(), num_elem
                )

    # Log validation results
    logger.info(f"Validation Epoch {epoch} - Average Loss: {avg_val_loss:.4f}")
    logger.info("=" * 60)
    logger.info("VALIDATION METRICS SUMMARY:")
    logger.info("=" * 60)

    # Log overall metrics for each metric type
    for metric_name in metric_names:
        if metric_name == "CRPS":
            # Log CRPS only when it has been computed to avoid empty MetricTracker access.
            if compute_crps:
                final_avg_pred = val_metrics["average_pred_vs_fine_CRPS"].getmean()
                std_avg_pred = val_metrics["average_pred_vs_fine_CRPS"].getstd()

                logger.info("OVERALL CRPS:")
                logger.info(
                    f" └── Average Prediction vs Fine CRPS: {final_avg_pred:.5f} ± {std_avg_pred:.5f}"
                )
        else:
            final_avg_pred = val_metrics[
                f"average_pred_vs_fine_{metric_name}"
            ].getmean()
            final_avg_coarse = val_metrics[
                f"average_coarse_vs_fine_{metric_name}"
            ].getmean()
            std_avg_pred = val_metrics[f"average_pred_vs_fine_{metric_name}"].getstd()
            std_avg_coarse = val_metrics[
                f"average_coarse_vs_fine_{metric_name}"
            ].getstd()

            logger.info(f"OVERALL {metric_name} METRICS:")
            logger.info(
                f" └── Average Prediction vs Fine {metric_name}: {final_avg_pred:.4f} ± {std_avg_pred:.4f}"
            )
            logger.info(
                f" └── Average Coarse vs Fine {metric_name}: {final_avg_coarse:.4f} ± {std_avg_coarse:.4f}"
            )
            logger.info("")

    # Log per-variable metrics
    logger.info("PER-VARIABLE METRICS:")
    for var_name in args.varnames_list:
        logger.info(f" └── {var_name}:")
        for metric_name in metric_names:
            if metric_name == "CRPS":
                # Log CRPS only when it has been computed to avoid empty MetricTracker access.
                if compute_crps:
                    crps_var = val_metrics[f"{var_name}_pred_vs_fine_CRPS"].getmean()
                    crps_std = val_metrics[f"{var_name}_pred_vs_fine_CRPS"].getstd()
                    logger.info("   └── CRPS:")
                    logger.info(
                        f"       └── Model Pred vs Fine: {crps_var:.5f} ± {crps_std:.5f}"
                    )
            else:
                pred_metric = val_metrics[
                    f"{var_name}_pred_vs_fine_{metric_name}"
                ].getmean()
                pred_std = val_metrics[
                    f"{var_name}_pred_vs_fine_{metric_name}"
                ].getstd()

                coarse_metric = val_metrics[
                    f"{var_name}_coarse_vs_fine_{metric_name}"
                ].getmean()
                coarse_std = val_metrics[
                    f"{var_name}_coarse_vs_fine_{metric_name}"
                ].getstd()

                logger.info(f"   └── {metric_name}:")
                logger.info(
                    f"       └── Model Pred vs Fine: {pred_metric:.4f} ± {pred_std:.4f}"
                )
                logger.info(
                    f"       └── Coarse vs Fine:     {coarse_metric:.4f} ± {coarse_std:.4f}"
                )

    # To verify with Kazem
    # Global heatmap of validation metrics (per variable × metric)
    if paths is not None:
        try:
            heatmap_path = plot_metrics_heatmap(
                valid_metrics_history=val_metrics,
                variable_names=args.varnames_list,
                metric_names=metric_names,
                filename=f"{args.run_type}_validation_metrics_epoch_{epoch}",
                save_dir=paths.results,
            )
            logger.info(f"Saved validation metrics heatmap to: {heatmap_path}")
        except Exception as e:
            logger.warning(f"Could not generate metrics heatmap: {e}")

    # Check if we should create plots for this batch
    should_plot = (
        plot_every_n_epochs is not None
        and epoch % plot_every_n_epochs == 0
        and paths is not None
    )
    if should_plot:
        logger.info("Reconstructing and plots ...")

        _ = reconstruct_original_layout(
            epoch,
            args,
            paths,
            steps,
            all_data=all_data,
            dataset=valid_dataset,
            # device=device,  # Keep on the same device --> OOM
            device=torch.device(
                "cpu"
            ),  # reconstruction & plotting on CPU to avoid cuda out of memory
            logger=logger,  # Pass the logger
        )
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)

        # Log overall metrics for each metric type
        for metric_name in metric_names:
            if metric_name == "CRPS":
                # Log CRPS only when it has been computed to avoid empty MetricTracker access.
                if compute_crps:
                    final_avg_pred = val_metrics["average_pred_vs_fine_CRPS"].getmean()
                    std_pred = val_metrics["average_pred_vs_fine_CRPS"].getstd()
                    writer.add_scalar(
                        "Metrics/average_pred_vs_fine_CRPS", final_avg_pred, epoch
                    )
                    writer.add_scalar(
                        "Metrics/average_pred_vs_fine_CRPS_std", std_pred, epoch
                    )
            else:
                final_avg_pred = val_metrics[
                    f"average_pred_vs_fine_{metric_name}"
                ].getmean()
                std_pred = val_metrics[f"average_pred_vs_fine_{metric_name}"].getstd()

                final_avg_coarse = val_metrics[
                    f"average_coarse_vs_fine_{metric_name}"
                ].getmean()
                std_coarse = val_metrics[
                    f"average_coarse_vs_fine_{metric_name}"
                ].getstd()
                writer.add_scalar(
                    f"Metrics/average_pred_vs_fine_{metric_name}", final_avg_pred, epoch
                )
                writer.add_scalar(
                    f"Metrics/average_pred_vs_fine_{metric_name}_std", std_pred, epoch
                )
                writer.add_scalar(
                    f"Metrics/average_coarse_vs_fine_{metric_name}",
                    final_avg_coarse,
                    epoch,
                )
                writer.add_scalar(
                    f"Metrics/average_coarse_vs_fine_{metric_name}_std",
                    std_coarse,
                    epoch,
                )

        # Log per-variable metrics
        for var_name in args.varnames_list:
            for metric_name in metric_names:
                if metric_name == "CRPS":
                    # Log CRPS only when it has been computed to avoid empty MetricTracker access.
                    if compute_crps:
                        crps_var = val_metrics[
                            f"{var_name}_pred_vs_fine_CRPS"
                        ].getmean()
                        crps_var_std = val_metrics[
                            f"{var_name}_pred_vs_fine_CRPS"
                        ].getstd()

                        writer.add_scalar(
                            f"Metrics/{var_name}_pred_vs_fine_CRPS", crps_var, epoch
                        )
                        writer.add_scalar(
                            f"Metrics/{var_name}_pred_vs_fine_CRPS_std",
                            crps_var_std,
                            epoch,
                        )

                else:
                    pred_metric = val_metrics[
                        f"{var_name}_pred_vs_fine_{metric_name}"
                    ].getmean()
                    pred_metric_std = val_metrics[
                        f"{var_name}_pred_vs_fine_{metric_name}"
                    ].getstd()

                    coarse_metric = val_metrics[
                        f"{var_name}_coarse_vs_fine_{metric_name}"
                    ].getmean()
                    coarse_metric_std = val_metrics[
                        f"{var_name}_coarse_vs_fine_{metric_name}"
                    ].getstd()

                    writer.add_scalar(
                        f"Metrics/{var_name}_pred_vs_fine_{metric_name}",
                        pred_metric,
                        epoch,
                    )
                    writer.add_scalar(
                        f"Metrics/{var_name}_pred_vs_fine_{metric_name}_std",
                        pred_metric_std,
                        epoch,
                    )

                    writer.add_scalar(
                        f"Metrics/{var_name}_coarse_vs_fine_{metric_name}",
                        coarse_metric,
                        epoch,
                    )
                    writer.add_scalar(
                        f"Metrics/{var_name}_coarse_vs_fine_{metric_name}_std",
                        coarse_metric_std,
                        epoch,
                    )

    return avg_val_loss, val_metrics
