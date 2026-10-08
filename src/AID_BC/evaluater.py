# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/


import numpy as np
from pathlib import Path

from AID_BC.diagnostics import (
    plot_power_spectra,
    plot_qq_quantiles,
    plot_validation_pdfs,
    plot_metrics_heatmap,
    plot_surface,
)


class MetricTracker:
    """Track summary statistics incrementally."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset tracked statistics."""
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values):
        """
        Update statistics with new values.

        Parameters
        ----------
        values : array-like
            Values to accumulate.
        """
        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return

        self.count += values.size
        self.sum += values.sum()
        self.sum_sq += np.square(values).sum()
        self.min = min(self.min, values.min())
        self.max = max(self.max, values.max())

    def getmean(self):
        """Return the mean."""
        if self.count == 0:
            return np.nan

        return self.sum / self.count

    def getstd(self):
        """Return the standard deviation."""
        if self.count == 0:
            return np.nan

        mean = self.getmean()
        variance = self.sum_sq / self.count - mean**2

        return np.sqrt(max(variance, 0.0))

    def getmin(self):
        """Return the minimum."""
        return np.nan if self.count == 0 else self.min

    def getmax(self):
        """Return the maximum."""
        return np.nan if self.count == 0 else self.max

    def getsqrtmean(self):
        """Return the square root of the mean."""
        return np.sqrt(self.getmean())


def summary_metrics(data, name, variable_name, ref=None):
    """
    Compute summary statistics for one dataset.

    Parameters
    ----------
    data : array-like
        Data to evaluate.
    name : str
        Dataset name.
    ref : array-like or None, optional
        Reference data used to compute MAE and RMSE.

    Returns
    -------
    dict
        Summary statistics.
    """
    if variable_name.lower() in ["tp", "var_tp", "precipitation"]:
        data = data * 1000.0

        if ref is not None:
            ref = np.asarray(ref) * 1000.0

    data_tracker = MetricTracker()
    data_tracker.update(data)

    row = {
        "dataset": name,
        "MEAN": data_tracker.getmean(),
        "STD": data_tracker.getstd(),
        "MIN": data_tracker.getmin(),
        "MAX": data_tracker.getmax(),
    }

    if ref is None:
        row["MAE"] = np.nan
        row["RMSE"] = np.nan
    else:
        error = np.asarray(data) - np.asarray(ref)

        mae_tracker = MetricTracker()
        mae_tracker.update(np.abs(error))

        mse_tracker = MetricTracker()
        mse_tracker.update(np.square(error))

        row["MAE"] = mae_tracker.getmean()
        row["RMSE"] = mse_tracker.getsqrtmean()

    return row


def print_metrics(variable_name, summaries, logger):
    """
    Print summary metrics for one variable.

    Parameters
    ----------
    variable_name : str
        Climate variable name.
    summaries : list of dict
        Summary statistics.
    logger : Logger
        Logger used to report metrics.
    """
    lines = [
        f"Evaluation metrics for {variable_name}",
        (
            f"{'Dataset':14s} "
            f"{'MAE':>8s} "
            f"{'RMSE':>8s} "
            f"{'MEAN':>8s} "
            f"{'STD':>8s} "
            f"{'MIN':>8s} "
            f"{'MAX':>8s}"
        ),
    ]

    for row in summaries:
        lines.append(
            f"{row['dataset']:14s} "
            f"{row['MAE']:8.4f} "
            f"{row['RMSE']:8.4f} "
            f"{row['MEAN']:8.4f} "
            f"{row['STD']:8.4f} "
            f"{row['MIN']:8.4f} "
            f"{row['MAX']:8.4f}"
        )

    logger.info("\n".join(lines))


def evaluate(
    reference,
    raw,
    corrected,
    variable_names,
    method,
    year,
    logger,
    run_name,
    results_dir="./results",
):
    """
    Evaluate bias-corrected CMIP6 data against ERA5.

    Metrics are computed separately for each climate variable, while
    multivariable diagnostic figures contain one subplot per variable.

    Parameters
    ----------
    reference : xarray.Dataset
        ERA5 reference data on the CMIP6 grid.
    raw : xarray.Dataset
        Raw CMIP6 application data.
    corrected : xarray.Dataset
        Bias-corrected CMIP6 data.
    variable_names : sequence of str
        Ordered climate variables to evaluate.
    method : str
        Bias-correction method name.
    year : int
        Evaluated application year.
    logger : Logger
        Logger used to report evaluation progress.
    run_name : str
        Name of the bias-correction run used to create the corresponding
        subdirectory inside results_dir.
    results_dir : str or pathlib.Path, default="./results"
        Root directory for evaluation results.
    """
    result_dir = Path(results_dir) / run_name

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(f"Evaluating {method.upper()} correction for year {year}")

    reference_arrays = []
    raw_arrays = []
    corrected_arrays = []

    lat_1d = reference["latitude"].values
    lon_1d = reference["longitude"].values

    for variable_name in variable_names:
        reference_variable = reference[variable_name]
        raw_variable = raw[variable_name]
        corrected_variable = corrected[variable_name]

        summaries = [
            summary_metrics(
                reference_variable,
                "ERA5",
                variable_name,
            ),
            summary_metrics(
                raw_variable,
                "CMIP6_raw",
                variable_name,
                ref=reference_variable,
            ),
            summary_metrics(
                corrected_variable,
                f"CMIP6_corr_{method.upper()}",
                variable_name,
                ref=reference_variable,
            ),
        ]

        print_metrics(
            variable_name=variable_name,
            summaries=summaries,
            logger=logger,
        )

        variable_dir = result_dir / variable_name

        plot_metrics_heatmap(
            summaries=summaries,
            metric_names=["MAE", "RMSE"],
            filename="error_metrics.png",
            save_dir=variable_dir,
        )

        plot_metrics_heatmap(
            summaries=summaries,
            metric_names=["MEAN", "STD", "MIN", "MAX"],
            filename="distribution_metrics.png",
            save_dir=variable_dir,
        )

        reference_arrays.append(reference_variable)
        raw_arrays.append(raw_variable)
        corrected_arrays.append(corrected_variable)

    # dlat = float(
    #    np.abs(reference["latitude"].values[1] - reference["latitude"].values[0])
    # )

    # dlon = float(
    #    np.abs(reference["longitude"].values[1] - reference["longitude"].values[0])
    # )

    plot_validation_pdfs(
        predictions=corrected_arrays,
        targets=reference_arrays,
        coarse_inputs=raw_arrays,
        variable_names=list(variable_names),
        filename="validation_pdfs.png",
        save_dir=str(result_dir),
        save_npz=True,
    )

    plot_power_spectra(
        predictions=corrected_arrays,
        targets=reference_arrays,
        coarse_inputs=raw_arrays,
        dlat=0.25,
        dlon=0.25,
        variable_names=list(variable_names),
        filename="power_spectra.png",
        save_dir=str(result_dir),
        save_npz=True,
    )

    plot_qq_quantiles(
        predictions=corrected_arrays,
        targets=reference_arrays,
        coarse_inputs=raw_arrays,
        variable_names=list(variable_names),
        quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
        filename="qq_quantiles.png",
        save_dir=str(result_dir),
        save_npz=True,
    )

    num_time_steps_to_plot = min(3, reference.sizes["time"])
    for time_idx in range(num_time_steps_to_plot):
        reference_single_time = [
            x.isel(time=slice(time_idx, time_idx + 1)) for x in reference_arrays
        ]

        raw_single_time = [
            x.isel(time=slice(time_idx, time_idx + 1)) for x in raw_arrays
        ]

        corrected_single_time = [
            x.isel(time=slice(time_idx, time_idx + 1)) for x in corrected_arrays
        ]

        save_path = plot_surface(
            predictions=corrected_single_time,
            targets=reference_single_time,
            coarse_inputs=raw_single_time,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=list(variable_names),
            filename=f"surface_time_{time_idx:03d}.png",
            save_dir=str(result_dir),
        )

        logger.info(f"Saved surface plot for time step {time_idx} to: {save_path}")

    logger.success(f"Evaluation completed for year {year}: {result_dir}")
