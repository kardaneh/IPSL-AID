# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ.setdefault(
    "CARTOPY_DATA_DIR",
    "/leonardo_work/EUHPC_D27_095/cartopy_data",
)
import unittest
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib as mpl
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mpltex
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd


# ---------------------------------------------
# COMPLETE MATPLOTLIB STYLE CONFIGURATION
# ---------------------------------------------
params = {
    # DPI & figure settings
    # "figure.dpi": 150,
    # "savefig.dpi": 300,
    # Fonts
    "font.family": "DejaVu Sans",
    "mathtext.rm": "arial",
    "font.size": 12,  # General font size (affects ax.text())
    "font.style": "normal",  # 'normal', 'italic', 'oblique'
    "font.weight": "normal",  # 'normal', 'bold', 'heavy', 'light', 'ultrabold', 'ultralight'
    "font.stretch": "normal",  # Font stretch
    # Line properties
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    # Axis labels and titles
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    # Tick settings
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 10,
    "legend.loc": "best",
    "legend.frameon": False,
    # Text properties
    "text.color": "black",  # Default text color
    "text.usetex": False,  # LaTeX rendering
    "text.hinting": "auto",  # Text hinting
    "text.antialiased": True,  # Text anti-aliasing
    "text.latex.preamble": "",  # LaTeX preamble
}


mpl.rcParams.update(params)

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================


class PlotConfig:
    """Central configuration for all plotting functions."""

    # General settings
    DEFAULT_SAVE_DIR = "./results"
    DEFAULT_FIGSIZE_MULTIPLIER = 4

    # Color schemes
    COLORMAPS = {
        "temperature": "rainbow",
        "temp": "rainbow",
        "2t": "rainbow",
        "zonal": "BrBG_r",
        "10u": "BrBG_r",
        "meridional": "BrBG_r",
        "10v": "BrBG_r",
        "tp": "Blues",
        "TP": "Blues",
        "precipitation": "Blues",
        "dewpoint": "rainbow",
        "d2m": "rainbow",
        "surface temperature": "rainbow",
        "st": "rainbow",
        "pressure": "viridis",
        "pres": "viridis",
        "humidity": "Greens",
        "humid": "Greens",
        "wind": "coolwarm",
        "speed": "coolwarm",
        "mae": "Reds",
        "default": "viridis",
    }

    # Fixed visualization ranges for error diagnostics
    FIXED_DIFF_RANGES = {
        "T2M": (-5.0, 5.0),  # K
        "temperature": (-5.0, 5.0),
        "2t": (-5.0, 5.0),
        "VAR_2T": (-5.0, 5.0),
        "U10": (-5.0, 5.0),  # m/s
        "10u": (-5.0, 5.0),
        "meridional": (-5.0, 5.0),
        "VAR_10U": (-5.0, 5.0),
        "V10": (-5.0, 5.0),  # m/s
        "10v": (-5.0, 5.0),
        "VAR_10V": (-5.0, 5.0),
        "TP": (-0.5, 0.5),  # mm/h
        "tp": (-0.5, 0.5),
        "VAR_TP": (-0.5, 0.5),
        "VAR_D2M": (-5.0, 5.0),  # K
        "VAR_ST": (-5.0, 5.0),  # K
    }

    FIXED_MAE_RANGES = {
        "T2M": (0.0, 3.0),
        "temperature": (0.0, 3.0),
        "2t": (0.0, 3.0),
        "VAR_2T": (0.0, 3.0),
        "U10": (0.0, 3.0),
        "10u": (0.0, 3.0),
        "meridional": (0.0, 3.0),
        "VAR_10U": (0.0, 3.0),
        "V10": (0.0, 3.0),
        "10v": (0.0, 3.0),
        "VAR_10V": (0.0, 3.0),
        "TP": (0.0, 1.0),
        "tp": (0.0, 1.0),
        "VAR_TP": (0.0, 1.0),
        "VAR_D2M": (0.0, 3.0),
        "VAR_ST": (0.0, 3.0),
    }

    # Geographic features
    COASTLINE_w = 0.5
    BORDER_w = 0.5
    LAKE_w = 0.5
    BORDER_STYLE = "--"

    # Colorbar settings
    COLORBAR_h = 0.02
    COLORBAR_PAD = 0.05

    @classmethod
    def get_colormap(cls, variable_name):
        """Get appropriate colormap for a variable."""
        var_lower = variable_name.lower()
        for key, cmap in cls.COLORMAPS.items():
            if key in var_lower:
                return cmap
        return cls.COLORMAPS["default"]

    @classmethod
    def get_plot_name(cls, variable_name):
        """Convert variable name to readable plot name."""
        # Remove common prefixes
        name = variable_name.replace("VAR_", "").replace("var_", "")

        # Special cases
        if name == "2T":
            return "Temperature [K]"
        elif name == "10U":
            return "Zonal Wind [m/s]"
        elif name == "10V":
            return "Meridional Wind [m/s]"
        elif name == "MSLP":
            return "Sea Level Pressure"
        elif name == "T2M":
            return "2m Temperature [K]"
        elif name == "U10":
            return "10m Zonal Wind [m/s]"
        elif name == "V10":
            return "10m Meridional Wind [m/s]"
        elif name == "TP":
            return "Precipitation [mm/h]"
        elif name == "tp":
            return "Precipitation [mm/h]"
        elif name == "D2M":
            return "Dewpoint [K]"
        elif name == "ST":
            return "Surface Temperature [K]"

        # General conversion
        name = name.replace("_", " ")
        return name.title()

    @classmethod
    def convert_units(cls, variable_name, data):
        """
        Safe unit conversion when required.
        - NEVER modifies input
        - Returns a new array only if conversion is needed
        """
        name = variable_name.lower()
        if name in ["tp", "var_tp", "precipitation"]:
            return data * 1000.0  # m to mm
        return data

    @staticmethod
    def get_fixed_diff_range(var_name):
        """Get fixed visualization range for signed differences (Prediction − Truth)."""
        return PlotConfig.FIXED_DIFF_RANGES.get(var_name, None)

    @staticmethod
    def get_fixed_mae_range(var_name):
        """Get fixed visualization range for Mean Absolute Error (MAE)."""
        return PlotConfig.FIXED_MAE_RANGES.get(var_name, None)


def plot_validation_hexbin(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    variable_names=None,  # List of variable names
    filename="validation_hexbin.png",
    save_dir="./results",
    figsize_multiplier=4,  # Base size per subplot
):
    """
    Create hexbin plots comparing model predictions vs ground truth for all variables.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor or np.array, optional
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots
    """

    # Convert to numpy if they're tensors
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if coarse_inputs is not None and hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, h, w = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(num_vars)]

    # Calculate grid dimensions
    ncols = num_vars
    nrows = (num_vars + ncols - 1) // ncols  # Ceiling division

    # Create figure
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_multiplier, figsize_multiplier)
    )  #

    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.set_box_aspect(1)

    # Handle single subplot case
    if num_vars == 1:
        axes = np.array([axes])
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Plot each variable
    max_count = 0
    for i, (var_name, ax) in enumerate(zip(variable_names, axes_flat)):
        if i >= num_vars:
            ax.set_visible(False)
            continue

        # Flatten the spatial dimensions
        # pred_flat = predictions[:, i, :, :].reshape(-1)
        # target_flat = targets[:, i, :, :].reshape(-1)

        pred_i = PlotConfig.convert_units(var_name, predictions[:, i])
        tgt_i = PlotConfig.convert_units(var_name, targets[:, i])

        pred_flat = pred_i.reshape(-1)
        target_flat = tgt_i.reshape(-1)

        # Create hexbin plot
        hb = ax.hexbin(
            target_flat, pred_flat, gridsize=100, cmap="jet", bins="log", mincnt=1
        )

        # Get counts for colorbar scaling
        counts = hb.get_array()
        if counts is not None:
            max_count = max(max_count, np.max(counts))

        # Add identity line
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)

        # Calculate metrics
        r2 = r2_score(target_flat, pred_flat)
        mae = np.mean(np.abs(pred_flat - target_flat))
        rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

        # Add metrics to plot
        textstr = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}"
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        # Set title
        # ax.set_title(f'{var_name}')
        plot_name = PlotConfig.get_plot_name(var_name)
        ax.set_title(plot_name)

        # Set equal aspect ratio
        # ax.set_aspect('equal')

        # Format ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # Only show y-label for leftmost subplots
        if i % ncols == 0:  # First column
            ax.set_ylabel("Predicted Values")
        else:
            ax.set_ylabel("")  # Remove y-label for non-leftmost plots

        # Only show x-label for bottom row subplots
        if i >= (nrows - 1) * ncols:  # Last row
            ax.set_xlabel("True Values")
        else:
            ax.set_xlabel("")  # Remove x-label for non-bottom plots

    # Add colorbar
    # cbar_width_per_subplot = 0.02
    # actual_cbar_width = cbar_width_per_subplot / num_vars
    # cbar_ax = fig.add_axes([0.92, 0.1, actual_cbar_width, 0.8])
    # cbar = fig.colorbar(hb, cax=cbar_ax, label=r"$\mathrm{\log_{10}[Count]}$")
    # Colorbar attached to the LAST axis
    ax_last = axes_flat[min(num_vars - 1, len(axes_flat) - 1)]

    cax = ax_last.inset_axes([1.05, 0.0, 0.04, 1.0])  # [x, y, width, height]
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label(r"$\log_{10}[\mathrm{Count}]$")

    plt.subplots_adjust(
        hspace=0.1, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_comparison_hexbin(
    predictions,
    targets,
    coarse_inputs,
    variable_names=None,
    filename="comparison_hexbin.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Create hexbin comparison plots between model predictions, ground truth, and coarse inputs.

    For each variable, creates two side-by-side hexbin plots:
    1. Model predictions vs ground truth
    2. Coarse inputs vs ground truth

    Each plot includes an identity line and R²/MAE metrics.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor or np.array
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles. If None, uses VAR_0, VAR_1, etc.
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    save_path : str
        Path to the saved figure
    """

    # Convert tensors → numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, h, w = predictions.shape

    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(num_vars)]

    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]

    # For color scaling: collect all hexbin counts
    all_counts = []

    # ----------------------------------------
    # 2) Pre-pass: collect hexbin densities
    # ----------------------------------------
    # for i in range(num_vars):
    # target_flat = targets[:, i].reshape(-1)
    # pred_flat   = predictions[:, i].reshape(-1)
    # coarse_flat = coarse_inputs[:, i].reshape(-1)

    for i, var_name in enumerate(variable_names):
        pred_i = PlotConfig.convert_units(var_name, predictions[:, i])
        tgt_i = PlotConfig.convert_units(var_name, targets[:, i])
        coarse_i = PlotConfig.convert_units(var_name, coarse_inputs[:, i])

        pred_flat = pred_i.reshape(-1)
        target_flat = tgt_i.reshape(-1)
        coarse_flat = coarse_i.reshape(-1)

        # Use a temporary invisible axes to get density arrays
        fig_tmp, ax_tmp = plt.subplots()

        hb1 = ax_tmp.hexbin(
            target_flat, pred_flat, gridsize=100, cmap="jet", bins="log", mincnt=1
        )
        hb2 = ax_tmp.hexbin(
            target_flat, coarse_flat, gridsize=100, cmap="jet", bins="log", mincnt=1
        )

        all_counts.append(hb1.get_array())
        all_counts.append(hb2.get_array())

        plt.close(fig_tmp)

    # Global colorbar limits
    all_counts = np.concatenate(all_counts)
    global_vmin = np.min(all_counts)
    global_vmax = np.max(all_counts)

    # ----------------------------------------
    # 3) Actual plot
    # ----------------------------------------
    fig, axes = plt.subplots(
        num_vars,
        2,
        figsize=(2 * figsize_multiplier, num_vars * figsize_multiplier * 0.8),
    )

    plt.subplots_adjust(
        hspace=0.3, wspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    if num_vars == 1:
        axes = axes.reshape(1, -1)

    last_hb = None

    for i, var_name in enumerate(variable_names):
        # target_flat = targets[:, i].reshape(-1)
        # pred_flat   = predictions[:, i].reshape(-1)
        # coarse_flat = coarse_inputs[:, i].reshape(-1)

        pred_i = PlotConfig.convert_units(var_name, predictions[:, i])
        tgt_i = PlotConfig.convert_units(var_name, targets[:, i])
        coarse_i = PlotConfig.convert_units(var_name, coarse_inputs[:, i])

        pred_flat = pred_i.reshape(-1)
        target_flat = tgt_i.reshape(-1)
        coarse_flat = coarse_i.reshape(-1)

        # Calculate per-variable min/max for this variable
        var_min = min(target_flat.min(), pred_flat.min(), coarse_flat.min())
        var_max = max(target_flat.max(), pred_flat.max(), coarse_flat.max())

        # Add a small margin
        margin = 0.05 * (var_max - var_min)
        plot_min = var_min - margin
        plot_max = var_max + margin

        # --------------------------
        # Left: Model vs True
        # --------------------------
        ax = axes[i, 0]
        hb = ax.hexbin(
            target_flat,
            pred_flat,
            gridsize=100,
            cmap="jet",
            bins="log",
            mincnt=1,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        last_hb = hb  # store for colorbar

        # Use per-variable axis limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        # identity line
        ax.plot([plot_min, plot_max], [plot_min, plot_max], "r--", alpha=0.7)

        r2 = r2_score(target_flat, pred_flat)
        mae = np.mean(np.abs(pred_flat - target_flat))
        ax.text(
            0.05,
            0.95,
            f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}",
            transform=ax.transAxes,
            va="top",
        )

        ax.set_title(f"{plot_variable_names[i]} – Model vs True")
        ax.set_ylabel("Model Values")
        if i == num_vars - 1:
            ax.set_xlabel("True Values")
        else:
            ax.set_xlabel("")

        # --------------------------
        # Right: Coarse vs True
        # --------------------------
        ax = axes[i, 1]
        hb = ax.hexbin(
            target_flat,
            coarse_flat,
            gridsize=100,
            cmap="jet",
            bins="log",
            mincnt=1,
            vmin=global_vmin,
            vmax=global_vmax,
        )
        last_hb = hb

        # Use the same per-variable limits
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        ax.plot(
            [plot_min, plot_max], [plot_min, plot_max], "r--", alpha=0.7, linewidth=1
        )

        r2 = r2_score(target_flat, coarse_flat)
        mae = np.mean(np.abs(coarse_flat - target_flat))
        ax.text(
            0.05,
            0.95,
            f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}",
            transform=ax.transAxes,
            va="top",
        )

        ax.set_title(f"{plot_variable_names[i]} – Coarse vs True")
        ax.set_ylabel("Coarse Values")
        if i == num_vars - 1:
            ax.set_xlabel("True Values")
        else:
            ax.set_xlabel("")

    # ----------------------------------------
    # 4) Single shared colorbar
    # ----------------------------------------
    cbar_ax = fig.add_axes([0.98, 0.1, 0.02, 0.8])
    fig.colorbar(last_hb, cax=cbar_ax, label=r"$\log_{10}[\mathrm{Count}]$")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_metric_histories(
    valid_metrics_history,
    variable_names,
    metric_names,
    filename="validation_metrics",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Creates row-based panel plots: one figure per metric, rows = variables, shared x-axis.

    Parameters
    ----------
    valid_metrics_history : dict
        Dict from training loop storing metric histories.
    variable_names : list of str
        Names of variables.
    metric_names : list of str
        List of metric names (e.g. ["MAE"]).
    filename : str
        Prefix for saved figures.
    save_dir : str
        Directory where images are saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    num_vars = len(variable_names)

    for metric in metric_names:
        # Rows = variables, 1 column, shared x-axis
        fig, axes = plt.subplots(
            nrows=num_vars,
            ncols=1,
            figsize=(6, figsize_multiplier * num_vars),
            squeeze=False,
            sharex=True,
        )
        plt.subplots_adjust(hspace=0.1, left=0.15, right=0.95, top=0.9, bottom=0.1)

        for i, var in enumerate(variable_names):
            ax = axes[i, 0]

            key_pred = f"{var}_pred_vs_fine_{metric}"
            key_coarse = f"{var}_coarse_vs_fine_{metric}"

            if (
                key_pred not in valid_metrics_history
                or key_coarse not in valid_metrics_history
            ):
                ax.text(0.5, 0.5, "Missing Data", ha="center", va="center")
                ax.set_yscale("log")
                continue

            pred_hist = valid_metrics_history[key_pred]
            coarse_hist = valid_metrics_history[key_coarse]

            # Plot
            linestyles = mpltex.linestyle_generator(markers=[])
            ax.plot(pred_hist, label="Pred vs Fine", **next(linestyles))
            ax.plot(coarse_hist, label="Coarse vs Fine", **next(linestyles))

            ax.set_yscale("log")
            # ax.set_ylabel(rf"$\mathrm{{{metric}\ ({var})}}$")
            ax.set_ylabel(f"{metric} ({var})")

            ax.grid(True, alpha=0.3)
            ax.legend()

            # Only bottom row shows x-axis label
            if i == num_vars - 1:
                ax.set_xlabel("Epoch")
            else:
                ax.tick_params(labelbottom=False)

        save_path = os.path.join(save_dir, f"{filename}_{metric}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path


def plot_metrics_heatmap(
    valid_metrics_history,
    variable_names,
    metric_names,
    filename="validation_metrics_heatmap",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Plot a heatmap of validation metrics.

    Parameters
    ----------
    valid_metrics_history : dict
        Dict from validation loop storing metric histories.
    variable_names : list of str
        Names of variables.
    metric_names : list of str
        List of metric names (["MAE", "NMAE", "RMSE", "R²"]).
    filename : str
        Prefix for saved figures.
    save_dir : str
        Directory where images are saved.
    figsize_multiplier : float
        Controls overall figure size
    """

    os.makedirs(save_dir, exist_ok=True)

    # Build DataFrame
    data = {}

    for metric in metric_names:
        values = []

        for var in variable_names:
            key = f"{var}_pred_vs_fine_{metric}"

            if key in valid_metrics_history:
                tracker = valid_metrics_history[key]

                if tracker.count > 0:
                    value = tracker.getmean()

                    # Convert only dimensional metrics
                    if metric.lower() in ["mae", "rmse"]:
                        value = PlotConfig.convert_units(var, value)
                else:
                    value = np.nan
            else:
                value = np.nan

            values.append(value)

        data[metric] = values

    df = pd.DataFrame(data, index=variable_names)

    fig_width = figsize_multiplier + len(metric_names)
    fig_height = 0.6 * len(variable_names) + figsize_multiplier / 2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        df, ax=ax, cmap="viridis", annot=True, fmt=".2f", linewidths=0.8, cbar=True
    )

    ax.set_title("Validation metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Variable")

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_loss_histories(
    train_loss_history,
    valid_loss_history,
    filename="training_validation_loss.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Plots training and validation loss in a single panel.

    Parameters:
    -----------
    train_loss_history : list or array
        History of training loss values.
    valid_loss_history : list or array
        History of validation loss values.
    filename : str
        Output image file name for the plot.
    save_dir : str
        Directory to save the plot.
    """

    # Ensure inputs are lists
    if not isinstance(train_loss_history, list):
        train_loss_history = list(train_loss_history)
    if not isinstance(valid_loss_history, list):
        valid_loss_history = list(valid_loss_history)

    fig = plt.figure(figsize=(6, figsize_multiplier))
    ax = fig.add_subplot(111)

    epochs = range(len(train_loss_history))

    # Plot losses
    linestyles = mpltex.linestyle_generator(markers=[])
    ax.plot(epochs, train_loss_history, label="Training Loss", **next(linestyles))
    if valid_loss_history and any(valid_loss_history):
        ax.plot(epochs, valid_loss_history, label="Validation Loss", **next(linestyles))

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
    print(f"Loss history plot saved to: '{save_path}'")


def plot_average_metrics(
    valid_metrics_history,
    metric_names,  # List of metrics to plot
    filename="average_metrics.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Plots average metrics across all variables in a row-based layout with shared x-axis.

    Each row corresponds to one metric, plotting both:
        - average_pred_vs_fine_<metric>
        - average_coarse_vs_fine_<metric>

    Parameters
    ----------
    valid_metrics_history : dict
        Dictionary containing validation metrics history.
    metric_names : list of str
        Names of metrics to plot.
    filename : str
        Output image file name for the plot.
    save_dir : str
        Directory to save the plot.
    """
    if not metric_names:
        print("No metric names provided")
        return

    num_rows = len(metric_names)

    # Create figure: rows = num_rows, 1 column, share x-axis
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=1,
        figsize=(6, figsize_multiplier * num_rows),
        squeeze=False,
        sharex=True,
    )

    plt.subplots_adjust(hspace=0.1, left=0.15, right=0.95, top=0.95, bottom=0.1)

    for idx, metric in enumerate(metric_names):
        ax = axes[idx, 0]
        linestyles = mpltex.linestyle_generator(markers=[])

        # Keys
        key_pred = f"average_pred_vs_fine_{metric}"
        key_coarse = f"average_coarse_vs_fine_{metric}"

        # Plot pred vs fine
        if key_pred in valid_metrics_history:
            hist = valid_metrics_history[key_pred]
            if not isinstance(hist, list):
                hist = list(hist)
            ax.plot(hist, label="Pred vs Fine", **next(linestyles))

        # Plot coarse vs fine
        if key_coarse in valid_metrics_history:
            hist = valid_metrics_history[key_coarse]
            if not isinstance(hist, list):
                hist = list(hist)
            ax.plot(hist, label="Coarse vs Fine", **next(linestyles))

        ax.set_yscale("log")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Only bottom row gets x-label
        if idx == num_rows - 1:
            ax.set_xlabel("Epoch")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_spatiotemporal_histograms(
    steps,
    tindex_lim,
    centers,
    tindices,
    mode="train",
    filename="average_metrics.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Plot two 2D hexagonal bin histograms showing spatial-temporal data coverage:
    latitude center vs temporal index and longitude center vs temporal index.

    This function visualizes the distribution of data samples across spatial
    (latitude/longitude) and temporal dimensions using hexagonal binning,
    which provides smoother density estimation compared to rectangular binning.

    Parameters
    ----------
    steps : EasyDict
        Dictionary containing coordinate dimensions and limits. Expected to have
        attributes 'latitude' (or 'lat') and 'longitude' (or 'lon') specifying
        the maximum spatial indices.
    tindex_lim : tuple
        Tuple of (min_time, max_time) specifying the temporal index limits.
    centers : list of tuples
        List of (lat_center, lon_center) coordinates for each data sample.
        Each center represents the spatial location of a data point.
    tindices : list or array-like
        List of temporal indices corresponding to each data sample.
        Should have the same length as 'centers'.
    mode : str
        Dataset mode identifier, typically "train" or "validation".
        Used for plot title and filename.
    save_dir : str
        Directory path where the plot will be saved.
        Directory will be created if it doesn't exist.
    filename : str, optional
        Optional prefix to prepend to the output filename.
        Default is empty string.

    Returns
    -------
    None
        The function saves the plot to disk and does not return any value.

    Notes
    -----
    - The function creates two side-by-side subplots:
      1. Latitude center index vs temporal index
      2. Longitude center index vs temporal index
    - Uses hexagonal binning (hexbin) for density visualization, which reduces
      visual artifacts compared to rectangular histograms.
    - A single colorbar is shared between both plots with log10 scaling.
    - The color scale is normalized to the maximum count across both histograms.
    - Hexagons with zero count (mincnt=1) are not displayed.

    Examples
    --------
    >>> steps = EasyDict({'latitude': 180, 'longitude': 360})
    >>> tindex_lim = (0, 1000)
    >>> centers = [(10, 20), (15, 25), (10, 20), ...]  # list of (lat, lon)
    >>> tindices = [0, 5, 10, 15, ...]  # corresponding temporal indices
    >>> plot_spatiotemporal_histograms(steps, tindex_lim, centers,
    ...                                tindices, "train", "./plots")

    The function will save a plot named "spatiotemporal_train_hexbin.png"
    in the "./plots" directory.
    """
    if not centers or not tindices:
        print(f"No data to plot for {mode} mode")
        return

    # Convert to numpy arrays for efficient processing
    centers = np.array(centers)
    lat_centers = centers[:, 0]
    lon_centers = centers[:, 1]
    tindices = np.array(tindices)

    # Extract spatial limits from steps dictionary with fallback options
    max_lat = getattr(steps, "latitude", getattr(steps, "lat", None))
    max_lon = getattr(steps, "longitude", getattr(steps, "lon", None))
    min_time, max_time = tindex_lim

    # Create figure with two side-by-side subplots sharing y-axis
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(2 * figsize_multiplier, figsize_multiplier), sharey=True
    )
    plt.subplots_adjust(
        hspace=0.1, wspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    # Plot latitude vs time using hexagonal binning
    hex1 = ax1.hexbin(
        lat_centers,
        tindices,
        gridsize=100,  # Number of hexagons in x-direction
        extent=[0, max_lat, min_time, max_time],  # Data limits
        cmap="jet",  # Color map (assumed to be defined)
        mincnt=1,  # Only show hexagons with at least 1 count
        edgecolors="none",
    )  # No borders on hexagons
    ax1.set_xlabel("Latitude Center Index", fontsize=12)
    ax1.set_ylabel("Temporal Index", fontsize=12)
    ax1.set_xlim(0, max_lat)
    ax1.set_ylim(min_time, max_time)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Plot longitude vs time using hexagonal binning
    hex2 = ax2.hexbin(
        lon_centers,
        tindices,
        gridsize=100,
        extent=[0, max_lon, min_time, max_time],
        cmap="jet",
        mincnt=1,
        edgecolors="none",
    )
    ax2.set_xlabel("Longitude Center Index", fontsize=12)
    ax2.set_xlim(0, max_lon)
    ax2.set_ylim(min_time, max_time)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Normalize color scale to maximum count across both plots
    max_count = 1
    if hex1.get_array() is not None and len(hex1.get_array()) > 0:
        max_count = max(max_count, hex1.get_array().max())
    if hex2.get_array() is not None and len(hex2.get_array()) > 0:
        max_count = max(max_count, hex2.get_array().max())

    hex1.set_clim(0, max_count)
    hex2.set_clim(0, max_count)

    # Add single colorbar for both plots
    cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    fig.colorbar(hex1, cax=cbar_ax, label=r"$\log_{10}[\mathrm{Count}]$")

    # Save figure to disk
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{filename}spatiotemporal_{mode}_hexbin.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_surface(
    predictions,
    targets,
    coarse_inputs,
    lat_1d,
    lon_1d,
    timestamp=None,
    variable_names=None,
    filename="forecast_plot.png",
    save_dir=None,
    figsize_multiplier=None,
):
    """
    Plot side-by-side forecast maps (coarse_inputs input, true target, model prediction, and difference)
    for one or more meteorological variables over a geographic domain.

    Parameters
    ----------
    coarse_inputs : torch.Tensor or np.ndarray
        coarse_inputs-resolution input data with shape [1, n_vars, H, W].
    targets : torch.Tensor or np.ndarray
        Ground-truth high-resolution data with shape [1, n_vars, H, W].
    predictions : torch.Tensor or np.ndarray
        Model predictions at targets resolution with shape [1, n_vars, H, W].
    lat_1d : array-like
        1D array of latitude coordinates with shape [H].
    lon_1d : array-like
        1D array of longitude coordinates with shape [W].
    timestamp : datetime.datetime
        Forecast timestamp to include in the plot title.
    variable_names : list of str, optional
        Variable names or identifiers.
    filename : str, optional
        Output filename for saving the plot.
    save_dir : str, optional
        Directory to save the plot.
    figsize_multiplier : int, optional
        Base size multiplier for subplots.

    Returns
    -------
    None
    """

    # Use defaults from config if not provided
    if save_dir is None:
        save_dir = PlotConfig.DEFAULT_SAVE_DIR
    if figsize_multiplier is None:
        figsize_multiplier = PlotConfig.DEFAULT_FIGSIZE_MULTIPLIER

    # Convert tensors to numpy if needed
    if hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(lat_1d, "detach"):
        lat_1d = lat_1d.detach().cpu().numpy()
    if hasattr(lon_1d, "detach"):
        lon_1d = lon_1d.detach().cpu().numpy()

    # Create 2D meshgrid from 1D coordinates
    lat_min, lat_max = lat_1d.min(), lat_1d.max()
    lon_min, lon_max = lon_1d.min(), lon_1d.max()

    # Shape
    h, w = coarse_inputs[0, 0].shape
    lat_block = np.linspace(lat_max, lat_min, h)
    lon_block = np.linspace(lon_min, lon_max, w)
    lat, lon = np.meshgrid(lat_block, lon_block, indexing="ij")

    # Projection center
    lon_center = float((lon_min + lon_max) / 2)

    # Check data dimensions
    n_vars = coarse_inputs.shape[1]
    if targets.shape[1] != n_vars:
        raise ValueError(
            f"targets data has {targets.shape[1]} variables but coarse_inputs has {n_vars}"
        )
    if predictions.shape[1] != n_vars:
        raise ValueError(
            f"predictions data has {predictions.shape[1]} variables but coarse_inputs has {n_vars}"
        )

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(n_vars)]

    # Derive plot names and colormaps
    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]
    cmaps = [PlotConfig.get_colormap(var) for var in variable_names]

    # Derive vmin/vmax from data for each variable (for coarse_inputs, truth, prediction)
    vmin_list = []
    vmax_list = []

    # Derive vmin/vmax for difference plots (signed difference)
    diff_vmin_list = []
    diff_vmax_list = []

    for i in range(n_vars):
        var_name = variable_names[i]

        coarse_i = PlotConfig.convert_units(var_name, coarse_inputs[0, i])
        target_i = PlotConfig.convert_units(var_name, targets[0, i])
        pred_i = PlotConfig.convert_units(var_name, predictions[0, i])

        # Get combined data range for this variable (coarse_inputs, truth, prediction)
        # all_data = np.concatenate([#
        #    coarse_inputs[0, i].flatten(),
        #    targets[0, i].flatten(),
        #    predictions[0, i].flatten()
        # ])
        all_data = np.concatenate(
            [coarse_i.flatten(), target_i.flatten(), pred_i.flatten()]
        )

        # Calculate vmin/vmax (using quantile approach like original function)
        all_data_flat = all_data[~np.isnan(all_data)]
        if len(all_data_flat) > 0:
            q_low, q_high = np.quantile(all_data_flat, [0.02, 0.98])
            vmin, vmax = float(q_low), float(q_high)
        else:
            vmin, vmax = -1, 1

        # Ensure vmin < vmax
        if vmin >= vmax:
            vmin, vmax = float(np.nanmin(all_data)), float(np.nanmax(all_data))

        vmin_list.append(vmin)
        vmax_list.append(vmax)

        # Calculate signed difference between prediction and truth
        fixed_range = PlotConfig.get_fixed_diff_range(var_name)
        diff_data = (predictions[0, i] - targets[0, i]).flatten()
        diff_data = diff_data[~np.isnan(diff_data)]

        if fixed_range is not None:
            diff_vmin, diff_vmax = fixed_range
        else:
            if len(diff_data) > 0:
                # For signed difference, we want symmetric range around 0
                max_abs_diff = np.max(np.abs(diff_data))
                diff_vmin = -max_abs_diff * 1.1  # Add 10% padding
                diff_vmax = max_abs_diff * 1.1  # Add 10% padding

                # If all differences are zero or very small
                if diff_vmax <= 0.001:
                    diff_vmin, diff_vmax = -0.1, 0.1
            else:
                diff_vmin, diff_vmax = -1, 1

        diff_vmin_list.append(diff_vmin)
        diff_vmax_list.append(diff_vmax)

    # Use fixed figure size instead of geo_ratio calculation
    # This ensures rectangular panels regardless of location
    base_width_per_panel = 4.5  # Same as original scale
    base_height_per_panel = 3.0  # Keep this as is

    fig_width = base_width_per_panel * n_vars
    fig_height = base_height_per_panel * 4  # 4 rows

    # Set up figure
    fig, axes = plt.subplots(
        4,
        n_vars,  # 4 rows, n_vars columns
        figsize=(fig_width, fig_height),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=lon_center)
        },  # ccrs.Mercator(central_longitude=lon_center)
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},  # Keep spacing
    )

    # Main title
    if timestamp is not None:
        #    fig.suptitle(
        #        f"Forecast for {timestamp.strftime('%Y-%m-%d %H:%M')}",
        #        fontsize=16, y=1.02
        #    )
        print(f"Forecast for {timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Define geographic features
    # coastline = cfeature.COASTLINE.with_scale('50m')
    # borders = cfeature.BORDERS.with_scale('50m')
    # lakes = cfeature.LAKES.with_scale('50m')

    # Plot each variable
    for col_idx in range(n_vars):
        # Data for this variable
        # coarse_inputs_data = coarse_inputs[0, col_idx, :, :]
        # targets_data = targets[0, col_idx, :, :]
        # pred_data = predictions[0, col_idx, :, :]
        var_name = variable_names[col_idx]
        # plot_name = plot_variable_names[col_idx]

        coarse_inputs_data = PlotConfig.convert_units(
            var_name, coarse_inputs[0, col_idx]
        )
        targets_data = PlotConfig.convert_units(var_name, targets[0, col_idx])
        pred_data = PlotConfig.convert_units(var_name, predictions[0, col_idx])

        diff_data = pred_data - targets_data  # Signed difference (pred - truth)

        # Store image objects for rows that need colorbars
        im_coar = None
        im_diff = None

        # Process all rows
        for row_idx in range(4):
            ax = axes[row_idx, col_idx]

            # Select data based on row
            if row_idx == 0:
                data = coarse_inputs_data
                vmin, vmax = vmin_list[col_idx], vmax_list[col_idx]
                cmap = cmaps[col_idx]
            elif row_idx == 1:
                data = targets_data
                vmin, vmax = vmin_list[col_idx], vmax_list[col_idx]
                cmap = cmaps[col_idx]
            elif row_idx == 2:
                data = pred_data
                vmin, vmax = vmin_list[col_idx], vmax_list[col_idx]
                cmap = cmaps[col_idx]
            else:  # row_idx == 3
                data = diff_data
                vmin, vmax = diff_vmin_list[col_idx], diff_vmax_list[col_idx]
                cmap = "RdBu_r"  # Diverging colormap for differences

            # Create the plot
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                shading="auto",
            )

            # Store image objects for rows that need colorbars
            if row_idx == 0:
                im_coar = im
            elif row_idx == 3:
                im_diff = im

            # Set extent and features
            # ax.set_global()
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            # ax.coastlines(linewidth=0.5)
            # ax.add_feature(borders, linewidth=0.5, linestyle="--", edgecolor="black")
            # ax.add_feature(lakes, linewidth=0.5, edgecolor="black", facecolor="none")
            ax.coastlines(linewidth=0.6)
            ax.add_feature(
                cfeature.BORDERS.with_scale("50m"),
                linewidth=0.9,
                linestyle="--",
                edgecolor="black",
                zorder=11,
            )
            ax.add_feature(
                cfeature.LAKES.with_scale("50m"),
                edgecolor="black",
                facecolor="none",
                linewidth=0.9,
                zorder=9,
            )
            # ax.set_aspect("auto")  # CRITICAL: This makes panels rectangular regardless of projection
            ax.set_xticks([])
            ax.set_yticks([])

        # Add colorbar for PREDICTION row (row 2)
        if im_coar is not None:
            ax_coar = axes[0, col_idx]
            # Position at top of panel: [x, y, width, height] where y > 1.0 places it above
            cax_top = ax_coar.inset_axes([0.1, 1.05, 0.8, 0.05])
            cbar = fig.colorbar(im_coar, cax=cax_top, orientation="horizontal")
            cbar.set_label(f"{plot_variable_names[col_idx]}")
            cax_top.xaxis.set_ticks_position("top")
            cax_top.xaxis.set_label_position("top")

        # Add colorbar for DIFFERENCE row (row 3)
        if im_diff is not None:
            ax_diff = axes[3, col_idx]
            cax_diff = ax_diff.inset_axes([0.1, -0.12, 0.8, 0.05])
            fig.colorbar(
                im_diff,
                cax=cax_diff,
                orientation="horizontal",
                label=f"Δ {plot_variable_names[col_idx]} (Pred - Truth)",
            )

    # Add row labels on the left side
    row_labels = ["Coarse", "Truth", "Prediction", "Pred - Truth"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            rotation="vertical",
            fontsize=12,
        )

    # Adjust layout - give more room at bottom for colorbars
    fig.subplots_adjust(
        top=0.90, bottom=0.25, left=0.10, right=0.95, wspace=0.1, hspace=0.15
    )

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_global_surface_robinson(
    predictions,
    targets,
    coarse_inputs,
    lat_1d,
    lon_1d,
    timestamp=None,
    variable_names=None,
    filename="global_robinson.png",
    save_dir=None,
    figsize_multiplier=None,
):
    """
    Plot coarse, truth, prediction and difference fields in Robinson projection.

    Parameters
    ----------
    coarse_inputs : torch.Tensor or np.ndarray
        coarse_inputs-resolution input data with shape [1, n_vars, H, W].
    targets : torch.Tensor or np.ndarray
        Ground-truth high-resolution data with shape [1, n_vars, H, W].
    predictions : torch.Tensor or np.ndarray
        Model predictions at targets resolution with shape [1, n_vars, H, W].
    lat_1d : array-like
        1D array of latitude coordinates with shape [H].
    lon_1d : array-like
        1D array of longitude coordinates with shape [W].
    timestamp : datetime.datetime
        Forecast timestamp to include in the plot title.
    variable_names : list of str, optional
        Variable names or identifiers.
    filename : str, optional
        Output filename for saving the plot.
    save_dir : str, optional
        Directory to save the plot.
    figsize_multiplier : int, optional
        Base size multiplier for subplots.

    Returns
    -------
    None
    """

    # Use defaults from config if not provided
    if save_dir is None:
        save_dir = PlotConfig.DEFAULT_SAVE_DIR
    if figsize_multiplier is None:
        figsize_multiplier = PlotConfig.DEFAULT_FIGSIZE_MULTIPLIER

    # Convert tensors to numpy if needed
    if hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(lat_1d, "detach"):
        lat_1d = lat_1d.detach().cpu().numpy()
    if hasattr(lon_1d, "detach"):
        lon_1d = lon_1d.detach().cpu().numpy()  #

    # Create 2D meshgrid from 1D coordinates
    lat_min, lat_max = lat_1d.min(), lat_1d.max()
    lon_min, lon_max = lon_1d.min(), lon_1d.max()

    # Shape
    h, w = coarse_inputs[0, 0].shape
    lat_block = np.linspace(lat_max, lat_min, h)
    lon_block = np.linspace(lon_min, lon_max, w)
    lat2d, lon2d = np.meshgrid(lat_block, lon_block, indexing="ij")

    lon2d = ((lon2d + 180) % 360) - 180  # normalize

    # Check data dimensions
    n_vars = coarse_inputs.shape[1]
    if targets.shape[1] != n_vars:
        raise ValueError(
            f"targets data has {targets.shape[1]} variables but coarse_inputs has {n_vars}"
        )
    if predictions.shape[1] != n_vars:
        raise ValueError(
            f"predictions data has {predictions.shape[1]} variables but coarse_inputs has {n_vars}"
        )

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(n_vars)]

    # Derive plot names and colormaps
    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]
    cmaps = [PlotConfig.get_colormap(var) for var in variable_names]

    # Derive vmin/vmax from data for each variable (for coarse_inputs, truth, prediction)
    vmin_list = []
    vmax_list = []

    # Derive vmin/vmax for difference plots (signed difference)
    diff_vmin_list = []
    diff_vmax_list = []

    for i in range(n_vars):
        # Get combined data range for this variable (coarse_inputs, truth, prediction)
        all_data = np.concatenate(
            [
                coarse_inputs[0, i].flatten(),
                targets[0, i].flatten(),
                predictions[0, i].flatten(),
            ]
        )

        # Calculate vmin/vmax (using quantile approach like original function)
        all_data_flat = all_data[~np.isnan(all_data)]
        if len(all_data_flat) > 0:
            q_low, q_high = np.quantile(all_data_flat, [0.02, 0.98])
            vmin, vmax = float(q_low), float(q_high)
        else:
            vmin, vmax = -1, 1

        # Ensure vmin < vmax
        if vmin >= vmax:
            vmin, vmax = float(np.nanmin(all_data)), float(np.nanmax(all_data))

        vmin_list.append(vmin)
        vmax_list.append(vmax)

        # Calculate signed difference between prediction and truth
        diff_data = (predictions[0, i] - targets[0, i]).flatten()
        diff_data = diff_data[~np.isnan(diff_data)]

        if len(diff_data) > 0:
            # For signed difference, we want symmetric range around 0
            max_abs_diff = np.max(np.abs(diff_data))
            diff_vmin = -max_abs_diff * 1.1  # Add 10% padding
            diff_vmax = max_abs_diff * 1.1  # Add 10% padding

            # If all differences are zero or very small
            if diff_vmax <= 0.001:
                diff_vmin, diff_vmax = -0.1, 0.1
        else:
            diff_vmin, diff_vmax = -1, 1

        diff_vmin_list.append(diff_vmin)
        diff_vmax_list.append(diff_vmax)

    # Set up figure
    fig, axes = plt.subplots(
        4,
        n_vars,  # 4 rows, n_vars columns
        figsize=(4.5 * n_vars, 3.2 * 4),
        subplot_kw={"projection": ccrs.Robinson()},
        gridspec_kw={"hspace": 0.12, "wspace": 0.05},
    )

    if n_vars == 1:
        axes = axes.reshape(4, 1)

    row_labels = ["Coarse", "Truth", "Prediction", "Pred − Truth"]

    # Plot each variable
    for col in range(n_vars):
        coarse = coarse_inputs[0, col]
        truth = targets[0, col]
        pred = predictions[0, col]
        diff = pred - truth

        data_rows = [coarse, truth, pred, diff]
        vmins = [vmin_list[col]] * 3 + [diff_vmin_list[col]]
        vmaxs = [vmax_list[col]] * 3 + [diff_vmax_list[col]]
        cmaps_row = [cmaps[col]] * 3 + ["RdBu_r"]

        for row in range(4):
            ax = axes[row, col]
            ax.set_global()

            # Create the plot
            im = ax.pcolormesh(
                lon2d,
                lat2d,
                data_rows[row],
                transform=ccrs.PlateCarree(),
                cmap=cmaps_row[row],
                vmin=vmins[row],
                vmax=vmaxs[row],
                shading="auto",
            )

            ax.coastlines(linewidth=0.9)
            ax.add_feature(
                cfeature.BORDERS.with_scale("50m"),
                linewidth=0.9,
                linestyle="--",
                edgecolor="black",
                zorder=11,
            )
            ax.add_feature(
                cfeature.LAKES.with_scale("50m"),
                edgecolor="black",
                facecolor="none",
                linewidth=0.9,
                zorder=9,
            )
            ax.set_xticks([])
            ax.set_yticks([])

            # if row == 0:
            # ax.set_title(plot_variable_names[col], fontsize=13)

            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    rotation=90,
                    fontsize=12,
                )

            # Colorbars
            if row == 0:
                # cax = ax.inset_axes([0.1, 1.02, 0.8, 0.05])
                cax = ax.inset_axes([0.1, 1.08, 0.8, 0.05])
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.set_label(plot_variable_names[col])
                cax.xaxis.set_ticks_position("top")
                cax.xaxis.set_label_position("top")

            if row == 3:
                cax = ax.inset_axes([0.1, -0.12, 0.8, 0.05])
                fig.colorbar(
                    im,
                    cax=cax,
                    orientation="horizontal",
                    label=f"Δ {plot_variable_names[col]} (Pred - Truth)",
                )

    if timestamp is not None:
        fig.suptitle(
            f"Global Robinson diagnostic – {timestamp.strftime('%Y-%m-%d %H:%M')}",
            fontsize=16,
            y=0.96,
        )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_MAE_map(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    lat_1d,
    lon_1d,
    timestamp=None,
    variable_names=None,
    filename="validation_mae_map.png",
    save_dir=None,
    figsize_multiplier=None,  # Base size per subplot
):
    """
    Plot spatial MAE maps averaged over all time steps:
    MAE(x, y) = mean_t(abs(prediction - target))

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    lat_1d : array-like
        1D array of latitude coordinates with shape [H].
    lon_1d : array-like
        1D array of longitude coordinates with shape [W].
    timestamp : datetime.datetime
        Forecast timestamp to include in the plot title.
    variable_names : list of str, optional
        Variable names or identifiers.
    filename : str, optional
        Output filename for saving the plot.
    save_dir : str, optional
        Directory to save the plot.
    figsize_multiplier : int, optional
        Base size multiplier for subplots.

    Returns
    -------
    None
    """

    if save_dir is None:
        save_dir = PlotConfig.DEFAULT_SAVE_DIR
    if figsize_multiplier is None:
        figsize_multiplier = PlotConfig.DEFAULT_FIGSIZE_MULTIPLIER

    # Convert tensors to numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(lat_1d, "detach"):
        lat_1d = lat_1d.detach().cpu().numpy()
    if hasattr(lon_1d, "detach"):
        lon_1d = lon_1d.detach().cpu().numpy()

    lat_min, lat_max = lat_1d.min(), lat_1d.max()
    lon_min, lon_max = lon_1d.min(), lon_1d.max()

    T, n_vars, h, w = predictions.shape

    lat_block = np.linspace(lat_max, lat_min, h)
    lon_block = np.linspace(lon_min, lon_max, w)
    lat, lon = np.meshgrid(lat_block, lon_block, indexing="ij")

    lon_center = float((lon_min + lon_max) / 2)

    if targets.shape[1] != n_vars:
        raise ValueError("targets and predictions must have same number of variables")

    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(n_vars)]

    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]
    # cmaps = [PlotConfig.get_colormap(var) for var in variable_names]
    cmaps = PlotConfig.get_colormap("mae")

    vmin_list, vmax_list = [], []

    # MAE averaged over time for color scaling
    for i in range(n_vars):
        # mae_data = np.mean(np.abs(predictions[:, i] - targets[:, i]), axis=0)
        pred_i = PlotConfig.convert_units(variable_names[i], predictions[:, i])
        tgt_i = PlotConfig.convert_units(variable_names[i], targets[:, i])

        mae_data = np.mean(np.abs(pred_i - tgt_i), axis=0)

        mae_flat = mae_data.flatten()
        mae_flat = mae_flat[~np.isnan(mae_flat)]

        fixed_range = PlotConfig.get_fixed_mae_range(variable_names[i])

        if fixed_range is not None:
            vmin, vmax = fixed_range
        else:
            if len(mae_flat) > 0:
                q_low, q_high = np.quantile(mae_flat, [0.02, 0.98])
                vmin, vmax = float(q_low), float(q_high)
            else:
                vmin, vmax = 0.0, 1.0

        if vmin >= vmax:
            vmin, vmax = float(np.nanmin(mae_flat)), float(np.nanmax(mae_flat))

        vmin_list.append(vmin)
        vmax_list.append(vmax)

    base_width_per_panel = 4.5
    base_height_per_panel = 3.0

    fig_width = base_width_per_panel * n_vars
    fig_height = base_height_per_panel

    fig, axes = plt.subplots(
        1,
        n_vars,
        figsize=(fig_width, fig_height),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=lon_center)
        },  # ccrs.Mercator(central_longitude=lon_center)
        gridspec_kw={"wspace": 0.1},
    )

    if n_vars == 1:
        axes = [axes]

    if timestamp is not None:
        fig.suptitle(
            f"MAE Map (time-averaged) - {timestamp.strftime('%Y-%m-%d %H:%M')}",
            fontsize=16,
            y=1.05,
        )

    for col_idx in range(n_vars):
        ax = axes[col_idx]

        pred_i = PlotConfig.convert_units(
            variable_names[col_idx], predictions[:, col_idx]
        )
        tgt_i = PlotConfig.convert_units(variable_names[col_idx], targets[:, col_idx])

        # MAE averaged over all time steps
        mae_data = np.mean(np.abs(pred_i - tgt_i), axis=0)

        im = ax.pcolormesh(
            lon,
            lat,
            mae_data,
            vmin=vmin_list[col_idx],
            vmax=vmax_list[col_idx],
            cmap=cmaps,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # ax.set_global()
        ax.coastlines(linewidth=0.6)
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.6,
            linestyle="--",
            edgecolor="black",
            zorder=11,
        )
        ax.add_feature(
            cfeature.LAKES.with_scale("50m"),
            edgecolor="black",
            facecolor="none",
            linewidth=0.6,
            zorder=9,
        )
        # ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])

        cax = ax.inset_axes([0.1, -0.15, 0.8, 0.05])
        fig.colorbar(
            im,
            cax=cax,
            orientation="horizontal",
            label=f"MAE {plot_variable_names[col_idx]}",
        )

    fig.subplots_adjust(top=0.85, bottom=0.25, left=0.08, right=0.95)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_validation_pdfs(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    variable_names=None,  # List of variable names
    filename="validation_pdfs.png",
    save_dir="./results",
    figsize_multiplier=4,  # Base size per subplot
):
    """
    Create PDF (Probability Density Function) plots comparing distributions of
    model predictions vs ground truth for all variables.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor or np.array, optional
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    None
        The function saves the plot to disk and does not return any value.

    Notes
    -----
    - Creates horizontal subplots (one per variable) showing PDFs
    - Each subplot shows up to 3 lines: Predictions, Ground Truth, and Coarse Inputs
    - Uses automatic color and linestyle cycling based on global matplotlib settings
    - Calculates and displays key statistics for each distribution
    - Handles both PyTorch tensors and numpy arrays

    Examples
    --------
    >>> predictions = np.random.randn(10, 3, 64, 64)  # 10 samples, 3 variables
    >>> targets = np.random.randn(10, 3, 64, 64)
    >>> plot_validation_pdfs(predictions, targets, variable_names=['Temp', 'Pres', 'Humid'])
    """
    # Convert to numpy if they're tensors
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if coarse_inputs is not None and hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, h, w = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"Variable {i + 1}" for i in range(num_vars)]

    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]

    # Calculate grid dimensions for horizontal layout
    ncols = num_vars
    nrows = 1  # Single row for horizontal layout

    # Create figure with horizontal subplots
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_multiplier, figsize_multiplier)
    )

    # Handle single subplot case
    if num_vars == 1:
        axes = np.array([axes])
    if axes.ndim == 0:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax in axes:
        ax.set_box_aspect(1)
    plt.subplots_adjust(
        hspace=0.1, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    # pdf_npz_data = {}

    # Plot PDF for each variable
    for i, (var_name, ax) in enumerate(zip(variable_names, axes)):
        if i >= num_vars:
            ax.set_visible(False)
            continue
        linestyles = mpltex.linestyle_generator(markers=[])
        # Flatten the spatial dimensions
        pred_i = PlotConfig.convert_units(var_name, predictions[:, i])
        tgt_i = PlotConfig.convert_units(var_name, targets[:, i])
        plot_name = plot_variable_names[i]

        pred_flat = pred_i.reshape(-1)
        target_flat = tgt_i.reshape(-1)

        # Collect all data for combined range
        all_data = [pred_flat, target_flat]
        if coarse_inputs is not None:
            # coarse_flat = coarse_inputs[:, i, :, :].flatten() #.mean(axis=0).reshape(-1)
            coarse_i = PlotConfig.convert_units(var_name, coarse_inputs[:, i])
            coarse_flat = coarse_i.reshape(-1)

            all_data.append(coarse_flat)

        # Calculate global range for consistent x-axis
        all_values = np.concatenate(all_data)
        data_min = np.percentile(all_values, 0.25)  # 0.5th percentile
        data_max = np.percentile(all_values, 99.5)  # 99.5th percentile
        data_range = data_max - data_min

        # Extend range slightly for better visualization
        x_min = data_min - 0.05 * data_range
        x_max = data_max + 0.05 * data_range

        # Create bins for PDF calculation
        n_bins = 100
        bins = np.linspace(x_min, x_max, n_bins + 1)

        # Small epsilon to avoid log(0)
        epsilon = 1e-12

        # Plot log PDFs
        # Plot predictions
        hist_pred, bin_edges = np.histogram(pred_flat, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        log_hist_pred = np.log10(hist_pred + epsilon)
        ax.plot(bin_centers, log_hist_pred, label="Pred", **next(linestyles))

        # Plot ground truth
        hist_target, _ = np.histogram(target_flat, bins=bins, density=True)
        log_hist_target = np.log10(hist_target + epsilon)
        ax.plot(bin_centers, log_hist_target, label="Truth", **next(linestyles))

        # Plot coarse inputs if available
        if coarse_inputs is not None:
            hist_coarse, _ = np.histogram(coarse_flat, bins=bins, density=True)
            log_hist_coarse = np.log10(hist_coarse + epsilon)
            ax.plot(bin_centers, log_hist_coarse, label="Coarse", **next(linestyles))

        # Calculate and display statistics
        stats_text = []

        # Predictions statistics
        pred_mean = np.mean(pred_flat)
        pred_std = np.std(pred_flat)
        stats_text.append(f"Predictions: μ={pred_mean:.3f}, σ={pred_std:.3f}")

        # Ground truth statistics
        target_mean = np.mean(target_flat)
        target_std = np.std(target_flat)
        stats_text.append(f"Ground Truth: μ={target_mean:.3f}, σ={target_std:.3f}")

        # Coarse statistics if available
        if coarse_inputs is not None:
            coarse_mean = np.mean(coarse_flat)
            coarse_std = np.std(coarse_flat)
            stats_text.append(f"Coarse: μ={coarse_mean:.3f}, σ={coarse_std:.3f}")

        # Calculate KL divergence between predictions and ground truth
        hist_pred_safe = hist_pred + epsilon
        hist_target_safe = hist_target + epsilon

        # Normalize to probability distributions
        hist_pred_safe = hist_pred_safe / np.sum(hist_pred_safe)
        hist_target_safe = hist_target_safe / np.sum(hist_target_safe)

        kl_divergence = np.sum(
            hist_target_safe * np.log(hist_target_safe / hist_pred_safe)
        )

        # Add KL divergence to statistics
        stats_text.append(f"KL Divergence: {kl_divergence:.4f}")

        # Calculate correlation coefficient
        correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        stats_text.append(f"Correlation: {correlation:.4f}")

        # Add statistics as text box
        # stats_str = '\n'.join(stats_text)
        # ax.text(0.5, 1.02, stats_str, transform=ax.transAxes,
        #        verticalalignment='bottom', horizontalalignment='center')

        # Log statistics instead of plotting them
        print(f"[PDF stats] {plot_name}")
        print(f"  Predictions: μ={pred_mean:.3f}, σ={pred_std:.3f}")
        print(f"  Ground Truth: μ={target_mean:.3f}, σ={target_std:.3f}")
        if coarse_inputs is not None:
            print(f"  Coarse: μ={coarse_mean:.3f}, σ={coarse_std:.3f}")
        print(f"  KL Divergence: {kl_divergence:.4f}")
        print(f"  Correlation: {correlation:.4f}")

        # ax.set_xlabel(f'{var_name}')
        ax.set_xlabel(plot_name)

        # Only show y-label for leftmost subplot
        if i == 0:
            # ax.set_ylabel('log₁₀(PDF)')
            ax.set_ylabel(r"$\log_{10}(\mathrm{PDF})$")

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add legend
        ax.legend()

        # Set consistent x-limits
        # ax.set_xlim(x_min, x_max)

        # Set y-limits for log plot (handle cases where log values might be very negative)
        y_min = min(log_hist_pred.min(), log_hist_target.min())
        if coarse_inputs is not None:
            y_min = min(y_min, log_hist_coarse.min())
        y_max = max(log_hist_pred.max(), log_hist_target.max())
        if coarse_inputs is not None:
            y_max = max(y_max, log_hist_coarse.max())

        # Add small margin to y-limits
        y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Use scientific notation for large ranges
        if data_range > 1000:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        # key = f"{var_name}__pdf__"

        # pdf_npz_data[key + "bin_centers"] = bin_centers
        # pdf_npz_data[key + "log_pred"] = log_hist_pred
        # pdf_npz_data[key + "log_truth"] = log_hist_target

        # pdf_npz_data[key + "mean_pred"] = pred_mean
        # pdf_npz_data[key + "std_pred"] = pred_std
        # pdf_npz_data[key + "mean_truth"] = target_mean
        # pdf_npz_data[key + "std_truth"] = target_std
        # pdf_npz_data[key + "kl"] = kl_divergence
        # pdf_npz_data[key + "corr"] = correlation

        # if coarse_inputs is not None:
        #    pdf_npz_data[key + "log_coarse"] = log_hist_coarse
        #    pdf_npz_data[key + "mean_coarse"] = coarse_mean
        #    pdf_npz_data[key + "std_coarse"] = coarse_std

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # npz_path = os.path.splitext(save_path)[0] + ".npz"
    # np.savez_compressed(npz_path, **pdf_npz_data)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_power_spectra(
    predictions,  # Model predictions
    targets,  # Ground truth
    dlat,  # Grid spacing in latitude (degrees)
    dlon,  # Grid spacing in longitude (degrees)
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    variable_names=None,  # List of variable names
    filename="power_spectra_physical.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Calculate and plot power spectra with proper physical wavenumbers.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, nh, nw]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, nh, nw]
    dlat : float
        Grid spacing in latitude (degrees)
    dlon : float
        Grid spacing in longitude (degrees)
    coarse_inputs : torch.Tensor or np.array, optional
        Coarse inputs of shape [batch_size, num_variables, nh, nw]
    variable_names : list of str, optional
        Names of the variable names for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    None
    """
    # Convert to numpy if they're tensors
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if coarse_inputs is not None and hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, nh, nw = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"Variable {i + 1}" for i in range(num_vars)]

    # plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]

    # Calculate wavenumbers
    # FFT frequencies are in cycles per grid spacing
    fft_freq_lat = np.fft.fftfreq(nh, d=dlat)  # cycles per degree in lat direction
    fft_freq_lon = np.fft.fftfreq(nw, d=dlon)  # cycles per degree in lon direction

    # Shift frequencies so zero is at center
    fft_freq_lat_shifted = np.fft.fftshift(fft_freq_lat)
    fft_freq_lon_shifted = np.fft.fftshift(fft_freq_lon)

    # Create 2D wavenumber grid
    k_lat, k_lon = np.meshgrid(fft_freq_lon_shifted, fft_freq_lat_shifted)

    # Calculate magnitude of wavenumber vector (in cycles/degree)
    k_mag = np.sqrt(k_lat**2 + k_lon**2)

    # Create bins for radial averaging
    max_k = np.min([np.max(np.abs(fft_freq_lat)), np.max(np.abs(fft_freq_lon))])
    k_bins = np.linspace(0, max_k, min(nh, nw) // 2)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    # Create figure
    ncols = num_vars
    nrows = 1  # 2  Two rows: one for 2D spectrum, one for 1D spectrum

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_multiplier, nrows * figsize_multiplier),
        squeeze=False,
    )  # nrows * figsize_multiplier
    plt.subplots_adjust(
        hspace=0.2, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    axes = axes.ravel()
    for ax in axes:
        ax.set_box_aspect(1)

    """
    # Handle single subplot case
    if num_vars == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif axes.ndim == 1:
        axes = axes.reshape(nrows, ncols)
    """
    # spectra_npz_data = {}

    # spectra_npz_data["__meta__dlat"] = dlat
    # spectra_npz_data["__meta__dlon"] = dlon
    # spectra_npz_data["__meta__variables"] = np.array(variable_names)

    # Process each variable
    for i, var_name in enumerate(variable_names):
        if i >= num_vars:
            continue
        linestyles = mpltex.linestyle_generator(markers=[])
        # plot_name = plot_variable_names[i]

        # Initialize arrays for averaged PSDs
        psd2d_pred_sum = np.zeros((nh, nw))
        psd2d_target_sum = np.zeros((nh, nw))
        if coarse_inputs is not None:
            psd2d_coarse_sum = np.zeros((nh, nw))

        # Process each sample in the batch
        for b in range(batch_size):
            # Predictions
            # field_pred = predictions[b, i]
            field_pred = PlotConfig.convert_units(var_name, predictions[b, i])
            psd2d_pred = calculate_psd2d_simple(field_pred)
            psd2d_pred_sum += psd2d_pred

            # Targets
            # field_target = targets[b, i]
            field_target = PlotConfig.convert_units(var_name, targets[b, i])
            psd2d_target = calculate_psd2d_simple(field_target)
            psd2d_target_sum += psd2d_target

            # Coarse inputs
            if coarse_inputs is not None:
                # field_coarse = coarse_inputs[b, i]
                field_coarse = PlotConfig.convert_units(var_name, coarse_inputs[b, i])
                psd2d_coarse = calculate_psd2d_simple(field_coarse)
                psd2d_coarse_sum += psd2d_coarse

        # Average over batch
        psd2d_pred_avg = psd2d_pred_sum / batch_size
        psd2d_target_avg = psd2d_target_sum / batch_size
        if coarse_inputs is not None:
            psd2d_coarse_avg = psd2d_coarse_sum / batch_size

        # Calculate 1D radial spectra
        psd1d_pred = radial_average_psd(psd2d_pred_avg, k_mag, k_bins)
        psd1d_target = radial_average_psd(psd2d_target_avg, k_mag, k_bins)
        if coarse_inputs is not None:
            psd1d_coarse = radial_average_psd(psd2d_coarse_avg, k_mag, k_bins)

        # key = f"{var_name}__spectra__"

        # spectra_npz_data[key + "k"] = k_centers
        # spectra_npz_data[key + "psd_pred"] = psd1d_pred
        # spectra_npz_data[key + "psd_truth"] = psd1d_target

        # if coarse_inputs is not None:
        #    spectra_npz_data[key + "psd_coarse"] = psd1d_coarse

        """
        # --- Plot 2D PSD (top row) ---
        ax_top = axes[0, i] if num_vars > 1 else axes[0]

        # Use k_lon and k_lat for the axes instead of lat/lon
        k_lon_min, k_lon_max = fft_freq_lon_shifted[0], fft_freq_lon_shifted[-1]
        k_lat_min, k_lat_max = fft_freq_lat_shifted[0], fft_freq_lat_shifted[-1]

        im = ax_top.imshow(np.log10(psd2d_pred_avg + 1e-12),
                          cmap=cmap_white_jet,
                          aspect='auto',
                          origin='lower',
                          extent=[k_lon_min, k_lon_max, k_lat_min, k_lat_max])

        #ax_top.set_title(f'{var_name}')
        ax_top.set_title(plot_name)

        # Only add y-axis label for leftmost column
        if i == 0:
            ax_top.set_ylabel(r'$\mathrm{k_{lat}}$ (cycles/°)')
        else:
            ax_top.set_ylabel('')
            # Remove y-axis tick labels for non-leftmost columns
            ax_top.tick_params(axis='y', labelleft=False)

        # Always show x-axis label
        ax_top.set_xlabel(r'$\mathrm{k_{lon}}$ (cycles/°)')

        # Add grid for better readability
        ax_top.grid(True, alpha=0.3, linestyle='--')

        # Add colorbar for the last column only
        if i == num_vars - 1:
            cax = ax_top.inset_axes([1.05, 0, 0.05, 1])  # [x, y, w, h] relative to axes
            cbar = plt.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_label('log₁₀(PSD)')
        """

        # --- Plot 1D Radial Spectrum (bottom row) ---
        # ax_bottom = axes[1, i] if num_vars > 1 else axes[1]
        ax_bottom = axes[i]

        # Plot all spectra
        ax_bottom.loglog(k_centers, psd1d_pred, label="Pred", **next(linestyles))
        ax_bottom.loglog(k_centers, psd1d_target, label="Truth", **next(linestyles))

        if coarse_inputs is not None:
            ax_bottom.loglog(
                k_centers, psd1d_coarse, label="Coarse", **next(linestyles)
            )

        # Only add y-axis label for leftmost column
        if i == 0:
            ax_bottom.set_ylabel("PSD(k)")
        else:
            ax_bottom.set_ylabel("")

        # Always show x-axis label
        ax_bottom.set_xlabel("Wavenumber k [cycles/°]")

        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3, which="both")

        # Set reasonable axis limits
        valid = (k_centers > 0) & (psd1d_target > 0)
        if np.any(valid):
            ax_bottom.set_xlim(k_centers[valid][0] * 0.8, k_centers[valid][-1] * 1.2)

            # Find y-range
            y_min = min(psd1d_pred[valid].min(), psd1d_target[valid].min())
            y_max = max(psd1d_pred[valid].max(), psd1d_target[valid].max())
            if coarse_inputs is not None:
                y_min = min(y_min, psd1d_coarse[valid].min())
                y_max = max(y_max, psd1d_coarse[valid].max())

            ax_bottom.set_ylim(y_min * 0.5, y_max * 2.0)

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # npz_path = os.path.splitext(save_path)[0] + ".npz"
    # np.savez_compressed(npz_path, **spectra_npz_data)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def calculate_psd2d_simple(field):
    """
    Simple 2D PSD calculation without preprocessing.
    """
    fft = np.fft.fft2(field)
    psd2d = np.abs(np.fft.fftshift(fft)) ** 2
    return psd2d


def radial_average_psd(psd2d, k_mag, k_bins):
    """
    Radially average 2D PSD using wavenumber magnitude.
    """
    # Flatten arrays
    k_flat = k_mag.flatten()
    psd_flat = psd2d.flatten()

    # Use binned_statistic for radial averaging
    psd1d, _, _ = stats.binned_statistic(
        k_flat, psd_flat, statistic="mean", bins=k_bins
    )

    # Multiply by area of annulus (2πkΔk) to get proper spectral density
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    delta_k = k_bins[1:] - k_bins[:-1]
    area = 2 * np.pi * k_centers * delta_k

    # Avoid division by zero
    valid = area > 0
    psd1d[valid] = psd1d[valid] * area[valid]

    return psd1d


def plot_qq_quantiles(
    predictions,  # Model predictions
    targets,  # Ground truth
    coarse_inputs,  # Coarse inputs
    variable_names=None,  # List of variable names
    units=None,  # List of units for each variable
    quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
    filename="qq_quantiles.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Create QQ-plats at different quantiles comparing model predictions and
    coarse inputs against ground truth.

    For each variable, plots quantiles of predictions and coarse inputs
    against quantiles of ground truth with a 1:1 reference line.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor or np.array
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles.
        If None, uses ["VAR_0", "VAR_1", ...]
    units : list of str, optional
        Units for each variable for axis labels.
        If None, uses empty strings.
    quantiles : list of float, optional
        Quantile values to plot (e.g., [0.90, 0.95, 0.975, 0.99, 0.995])
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    save_path : str
        Path to the saved figure
    """

    # Convert tensors → numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, h, w = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(num_vars)]

    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]

    # Default units if not provided
    if units is None:
        units = [""] * num_vars

    # Figure setup
    fig, axes = plt.subplots(
        1,
        num_vars,
        figsize=(num_vars * figsize_multiplier, figsize_multiplier),
        constrained_layout=True,
    )

    plt.subplots_adjust(
        hspace=0.2, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    if num_vars > 1:
        axes = axes.ravel()
    # Handle single subplot case
    else:
        axes = np.array([axes])

    for ax in axes:
        ax.set_box_aspect(1)

    # qq_npz_data = {}

    # qq_npz_data["__meta__variables"] = np.array(variable_names)
    # qq_npz_data["__meta__quantiles"] = np.array(quantiles)

    for i, var_name in enumerate(variable_names):
        linestyles = mpltex.linestyle_generator(lines=[])
        ax = axes[i]
        plot_name = plot_variable_names[i]

        # Flatten spatial dims and average over batch
        # target_vals = targets[:, i]
        # pred_vals = predictions[:, i]
        # coarse_vals = coarse_inputs[:, i]
        pred_vals = PlotConfig.convert_units(var_name, predictions[:, i])
        target_vals = PlotConfig.convert_units(var_name, targets[:, i])
        coarse_vals = PlotConfig.convert_units(var_name, coarse_inputs[:, i])

        # Compute quantiles
        qs_target = np.quantile(target_vals, quantiles)
        qs_pred = np.quantile(pred_vals, quantiles)
        qs_coarse = np.quantile(coarse_vals, quantiles)

        # key = f"{var_name}__qq__"

        # qq_npz_data[key + "quantiles"] = np.array(quantiles)
        # qq_npz_data[key + "truth"] = qs_target
        # qq_npz_data[key + "pred"] = qs_pred
        # qq_npz_data[key + "coarse"] = qs_coarse

        print(f"[QQ Quantiles] {plot_name}")
        for q, qt, qp, qc in zip(quantiles, qs_target, qs_pred, qs_coarse):
            print(f"  q={q:.3f} | Truth={qt:.4f} | Pred={qp:.4f} | Coarse={qc:.4f} ")

        # ---- Plot predicted quantiles ----
        for q_idx, q in enumerate(quantiles):
            ax.plot(
                qs_target[q_idx],
                qs_pred[q_idx],
                label=f"{q * 100:.1f}%",
                **next(linestyles),
            )

        # ---- Plot coarse quantiles ----
        ax.plot(
            qs_target,
            qs_coarse,
            c="black",
            marker="s",
            label="Coarse",
            linestyle="None",
        )

        # ---- 1:1 reference line ----
        # Calculate appropriate limits for this variable
        min_val = min(qs_target.min(), qs_pred.min(), qs_coarse.min())
        max_val = max(qs_target.max(), qs_pred.max(), qs_coarse.max())
        margin = 0.0
        plot_min = min_val - margin
        plot_max = max_val + margin

        ax.plot(
            [plot_min, plot_max], [plot_min, plot_max], "r--", alpha=0.7, label="1:1"
        )

        # Set axis limits
        # ax.set_xlim(plot_min, plot_max)
        # ax.set_ylim(plot_min, plot_max)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

        # Labels and formatting
        # ax.set_title(var_name)
        ax.set_title(plot_name)

        # Add unit to labels if provided
        unit_str = f" ({units[i]})" if units[i] else ""

        # Only add y-axis label for leftmost plot
        if i == 0:
            ax.set_ylabel(f"Predicted/Coarse quantiles{unit_str}")

        ax.set_xlabel(f"True quantiles{unit_str}")

        ax.grid(True, linestyle="--", alpha=0.3)

        # Add legend only for first subplot
        if i == 0:
            ax.legend()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # npz_path = os.path.splitext(save_path)[0] + ".npz"
    # np.savez_compressed(npz_path, **qq_npz_data)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def dry_frequency_map(array, threshold):
    """
    Compute spatial dry pixels proportion maps. Value of each pixel corresponds to the frequency of dry weather for this pixel.

    Parameters
    ----------
    array : torch.Tensor or np.array
        Model predictions of shape [batch_size, h, w]
    threshold : float
        threshold for precipitation (expressed in mm): under it, pixel is considered dry.
    Returns
    -------
    np.ndarray(np.float64) of shape [h,w]
    """
    # convert to numpy if tensor :
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    dry_array = (array < threshold).astype(np.float64)
    dry_array_map = np.mean(dry_array, axis=0)

    return dry_array_map


def plot_dry_frequency_map(
    predictions,  # Model predictions precipitation (fine predicted)
    targets,  # Ground truth precipitation (fine true)
    threshold,  # threshold to define dry and wet (in mm)
    lat_1d,
    lon_1d,
    filename="validation_dry_frequency_map.png",
    save_dir=None,
    figsize_multiplier=None,  # Base size per subplot
):
    """
    Plot spatial dry pixels proportion maps. Value of each pixel corresponds to the frequency of dry weather for this pixel.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, h, w]
    threshold : float
        threshold for precipitation (expressed in mm): under it, pixel is considered dry.
    lat_1d : array-like
        1D array of latitude coordinates with shape [H].
    lon_1d : array-like
        1D array of longitude coordinates with shape [W].
    filename : str, optional
        Output filename for saving the plot.
    save_dir : str, optional
        Directory to save the plot.
    figsize_multiplier : int, optional
        Base size multiplier for subplots.

    Returns
    -------
    None
    """
    if save_dir is None:
        save_dir = PlotConfig.DEFAULT_SAVE_DIR
    if figsize_multiplier is None:
        figsize_multiplier = PlotConfig.DEFAULT_FIGSIZE_MULTIPLIER

    # Convert tensors to numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if hasattr(lat_1d, "detach"):
        lat_1d = lat_1d.detach().cpu().numpy()
    if hasattr(lon_1d, "detach"):
        lon_1d = lon_1d.detach().cpu().numpy()

    lat_min, lat_max = lat_1d.min(), lat_1d.max()
    lon_min, lon_max = lon_1d.min(), lon_1d.max()

    _, h, w = targets.shape

    lat_block = np.linspace(lat_max, lat_min, h)
    lon_block = np.linspace(lon_min, lon_max, w)
    lat, lon = np.meshgrid(lat_block, lon_block, indexing="ij")

    lon_center = float((lon_min + lon_max) / 2)

    cmap = PlotConfig.get_colormap(
        "dry frequency"
    )  # need to define the comap in PlotConfig

    # convert units :
    predictions = PlotConfig.convert_units("precipitation", predictions)
    targets = PlotConfig.convert_units("precipitation", targets)

    dry_freq_pred_map = dry_frequency_map(predictions, threshold)
    # dry_freq_pred = np.mean(dry_freq_pred_map)

    dry_freq_targ_map = dry_frequency_map(targets, threshold)
    # dry_freq_targ = np.mean(dry_freq_targ_map)

    vmin = 0
    vmax = 1

    base_width_per_panel = 4.5
    base_height_per_panel = 3.0

    fig_width = 3 * base_width_per_panel
    fig_height = 3 * base_height_per_panel

    fig, axes = plt.subplots(
        3,
        figsize=(fig_width, fig_height),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=lon_center)
        },  # ccrs.Mercator(central_longitude=lon_center)
        gridspec_kw={"wspace": 0.1},
    )
    im = axes[0].pcolormesh(
        lon,
        lat,
        dry_freq_pred_map,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    axes[0].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    axes[0].coastlines(linewidth=0.6)
    axes[0].add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.6,
        linestyle="--",
        edgecolor="black",
        zorder=11,
    )
    axes[0].add_feature(
        cfeature.LAKES.with_scale("50m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.6,
        zorder=9,
    )
    # ax.set_aspect("auto")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title("Predicted")

    im = axes[1].pcolormesh(
        lon,
        lat,
        dry_freq_targ_map,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    axes[1].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    axes[1].coastlines(linewidth=0.6)
    axes[1].add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.6,
        linestyle="--",
        edgecolor="black",
        zorder=11,
    )
    axes[1].add_feature(
        cfeature.LAKES.with_scale("50m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.6,
        zorder=9,
    )
    # ax.set_aspect("auto")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Target")

    fig.colorbar(
        im,
        ax=axes[0:2],
        location="right",
        orientation="vertical",
        label="frequency",
    )

    # vmax_diff = max(
    #     np.abs(np.max(dry_freq_pred_map - dry_freq_targ_map)),
    #     np.abs(np.min(dry_freq_pred_map - dry_freq_targ_map)),
    # )
    # norm_diff = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax = 1)

    im = axes[2].pcolormesh(
        lon,
        lat,
        dry_freq_pred_map - dry_freq_targ_map,
        norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        cmap="seismic",
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    axes[2].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    axes[2].coastlines(linewidth=0.6)
    axes[2].add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.6,
        linestyle="--",
        edgecolor="black",
        zorder=11,
    )
    axes[2].add_feature(
        cfeature.LAKES.with_scale("50m"),
        edgecolor="black",
        facecolor="none",
        linewidth=0.6,
        zorder=9,
    )
    # ax.set_aspect("auto")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Predicted frequency - Target frequency")

    fig.colorbar(
        im,
        ax=axes[2],
        location="right",
        orientation="vertical",
        label="frequency",
    )

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def calculate_pearsoncorr_nparray(arr1, arr2, axis=0):
    """
    Calculate Pearson correlation between 2 N-dimensional numpy arrays.

    Parameters:
    -----------
    arr1 : numpy.ndarray
        First N-dimensional array
    arr2 : numpy.ndarray
        Second N-dimensional array (must have same shape as arr1)
    axis : int, default=0
        Axis along which to compute correlation

    Returns:
    --------
    numpy.ndarray
        Pearson correlation coefficients. Output has N-1 dimensions
        (input shape with the specified axis removed).

    """

    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Arrays must have the same shape. Got {arr1.shape} and {arr2.shape}"
        )

    if arr1.ndim < 2:
        raise ValueError(f"Arrays must be at least 2-dimensional. Got {arr1.ndim}D")

    if axis < 0:
        axis = arr1.ndim + axis

    if axis < 0 or axis >= arr1.ndim:
        raise ValueError(
            f"Axis {axis} is out of bounds for array of dimension {arr1.ndim}"
        )

    # Move the correlation axis to the front for easier processing
    arr1_moved = np.moveaxis(arr1, axis, 0)
    arr2_moved = np.moveaxis(arr2, axis, 0)

    # Reshape to 2D: (n_samples, n_features=nlat*nlon)
    n_samples = arr1_moved.shape[0]
    arr1_2d = arr1_moved.reshape(n_samples, -1)
    arr2_2d = arr2_moved.reshape(n_samples, -1)

    # Vectorized Pearson correlation computation
    # Center the data
    arr1_centered = arr1_2d - arr1_2d.mean(axis=0, keepdims=True)
    arr2_centered = arr2_2d - arr2_2d.mean(axis=0, keepdims=True)

    # Compute correlation (output has shape (n_features))
    numerator = (arr1_centered * arr2_centered).sum(axis=0)
    denominator = np.sqrt(
        (arr1_centered**2).sum(axis=0) * (arr2_centered**2).sum(axis=0)
    )

    # Avoid division by zero (set as 0.0 instead of inf or nan)
    correlations = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0
    )

    # Reshape back to original dimensions (without axis used for correlation)
    output_shape = list(
        arr1.shape
    )  # array shape in list format [n_samples, nlat, nlon]
    output_shape.pop(
        axis
    )  # removes axis dimension from list e.g. axis=0: -> [nlat, nlon]

    output_corr = (
        correlations.reshape(output_shape) if output_shape else correlations.item()
    )

    return output_corr


def plot_validation_mvcorr(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    lat,
    lon,
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    variable_names=None,  # List of variable names
    filename="validation_mvcorr.png",
    save_dir="./results",
    figsize_multiplier=4,  # Base size per subplot
):
    """
    Create multivariate correlation map plots comparing model predictions vs ground truth,
    for all combinations of variables.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    lat : array-like
        2D array of latitude coordinates with shape [h, w].
    lon : array-like
        2D array of longitude coordinates with shape [h, w].
    coarse_inputs : torch.Tensor or np.array, optional
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    save_path : str
        Path to the saved figure
    """

    # Convert to numpy if they're tensors
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if coarse_inputs is not None and hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()

    batch_size, num_vars, h, w = predictions.shape

    if num_vars < 2:
        print("ERROR: need at least 2 variables but num_vars < 2")
        return "0"

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(num_vars)]

    # Make list of tuples defining variable combinations
    list_var_combos = []
    for ii in range(num_vars - 1):
        for jj in range(num_vars - 1 - ii):
            list_var_combos.append((ii, ii + jj + 1))

    # Calculate grid dimensions
    ncols = 2
    if coarse_inputs is not None:
        ncols = 3
    nrows = int(num_vars * (num_vars - 1) / 2)  # no. distinct pairs of input variables
    aspect_ratio = 0.7
    fwidth = ncols * figsize_multiplier  # longitude range
    fheight = (
        nrows * figsize_multiplier * aspect_ratio * 1.02
    )  # latitude range + title and colorbar
    print("fwidth")
    print(fwidth)
    print("fheight")
    print(fheight)

    spa_cor_out = np.zeros([nrows, ncols - 1])
    spa_rmse_out = np.zeros([nrows, ncols - 1])

    # Set up figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fwidth, fheight),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        squeeze=False,
    )

    # Define geographic features
    coastline = cfeature.COASTLINE.with_scale("50m")
    borders = cfeature.BORDERS.with_scale("50m")
    # lakes = cfeature.LAKES.with_scale("50m")

    var_name_combo_list = []

    # Plot each combination of variables
    # max_count = 0
    for i, varComb in enumerate(list_var_combos):
        var_name_combo = variable_names[varComb[0]] + "_" + variable_names[varComb[1]]
        var_name_combo_list.append(var_name_combo)
        print(var_name_combo)

        # Compute Correlation
        pred_corr = calculate_pearsoncorr_nparray(
            predictions[:, varComb[0], :, :], predictions[:, varComb[1], :, :], axis=0
        )
        target_corr = calculate_pearsoncorr_nparray(
            targets[:, varComb[0], :, :], targets[:, varComb[1], :, :], axis=0
        )
        if coarse_inputs is not None:
            coarse_corr = calculate_pearsoncorr_nparray(
                coarse_inputs[:, varComb[0], :, :],
                coarse_inputs[:, varComb[1], :, :],
                axis=0,
            )

        spa_cor_out[i, 0] = np.corrcoef(
            pred_corr.reshape(pred_corr.size), target_corr.reshape(target_corr.size)
        )[0, 1]
        if coarse_inputs is not None:
            spa_cor_out[i, 1] = np.corrcoef(
                coarse_corr.reshape(coarse_corr.size),
                target_corr.reshape(target_corr.size),
            )[0, 1]

        spa_rmse_out[i, 0] = np.sqrt((np.square(pred_corr - target_corr)).mean())
        if coarse_inputs is not None:
            spa_rmse_out[i, 1] = np.sqrt((np.square(coarse_corr - target_corr)).mean())

        # Col 0: Truth
        ax_target = axes[i, 0]
        ax_target.pcolormesh(
            lon,
            lat,
            target_corr,
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax_target.add_feature(coastline, linewidth=PlotConfig.COASTLINE_w)
        ax_target.add_feature(
            borders,
            linewidth=PlotConfig.BORDER_w,
            edgecolor="black",
            linestyle=PlotConfig.BORDER_STYLE,
        )
        ax_target.set_aspect("auto")

        # Col 1: Prediction
        ax_pred = axes[i, 1]
        im_pred = ax_pred.pcolormesh(
            lon,
            lat,
            pred_corr,
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
        ax_pred.add_feature(coastline, linewidth=PlotConfig.COASTLINE_w)
        ax_pred.add_feature(
            borders,
            linewidth=PlotConfig.BORDER_w,
            edgecolor="black",
            linestyle=PlotConfig.BORDER_STYLE,
        )
        ax_pred.set_aspect("auto")

        if coarse_inputs is not None:
            # Col 2: Coarse input
            ax_coar = axes[i, 2]
            ax_coar.pcolormesh(
                lon,
                lat,
                coarse_corr,
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
                transform=ccrs.PlateCarree(),
                shading="auto",
            )
            ax_coar.add_feature(coastline, linewidth=PlotConfig.COASTLINE_w)
            ax_coar.add_feature(
                borders,
                linewidth=PlotConfig.BORDER_w,
                edgecolor="black",
                linestyle=PlotConfig.BORDER_STYLE,
            )
            ax_coar.set_aspect("auto")

    # Add col labels
    col_labels = ["Truth", "Prediction"]
    if coarse_inputs is not None:
        col_labels = ["Truth", "Prediction", "Coarse"]
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].text(
            0.5,
            1.2,
            label,
            transform=axes[0, col_idx].transAxes,
            va="top",
            ha="center",
            fontsize=20.0,
        )

    # Add row labels
    for row_idx, label in enumerate(var_name_combo_list):
        axes[row_idx, 0].text(
            -0.02,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            va="center",
            ha="right",
            rotation="vertical",
            fontsize=8.0,
        )

    # Add colorbar
    # cbar_width_per_subplot = 0.02
    # actual_cbar_width = cbar_width_per_subplot / ncols
    cbar_ax = fig.add_axes([0.0, -0.02, 1, 0.02])
    cbar = fig.colorbar(
        im_pred, cax=cbar_ax, label="correlation", orientation="horizontal"
    )
    cbar.ax.tick_params(labelsize=20)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    # _________________________________________
    # Output summary map statistics as heatmaps
    # Spatial Correlation and Spatial RMSE wrt target

    # Setupt axis labels
    xLabels = ["Prediction"]
    if coarse_inputs is not None:
        xLabels = ["Prediction", "Coarse"]

    yLabels = var_name_combo_list

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=((ncols + 2) * 2, 4)
    )  # , layout='constrained')

    sns.heatmap(
        spa_cor_out,
        ax=ax1,
        cbar=False,
        linewidth=0.5,
        annot=True,
        fmt=".3f",
        xticklabels=xLabels,
        yticklabels=yLabels,
        vmin=0.0,
        vmax=1.0,
        cmap=plt.get_cmap("Reds"),
    )
    fig.colorbar(
        ax1.collections[0],
        ax=ax1,
        location="left",
        use_gridspec=False,
        pad=0.1,
        label="correlation",
    )
    ax1.tick_params(axis="y", pad=90, length=0)
    ax1.tick_params(axis="x", length=0)
    ax1.yaxis.set_label_position("left")

    sns.heatmap(
        spa_rmse_out,
        ax=ax2,
        cbar=False,
        linewidth=0.5,
        annot=True,
        fmt=".3f",
        xticklabels=xLabels,
        yticklabels=[""] * ncols,
        vmin=0.0,
        vmax=0.3,
        cmap=plt.get_cmap("Reds_r"),
    )
    fig.colorbar(
        ax2.collections[0],
        ax=ax2,
        location="right",
        use_gridspec=False,
        pad=0.1,
        label="RMSE",
    )
    ax2.tick_params(rotation=0, length=0)
    ax2.yaxis.set_label_position("right")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    filenameCR = "SpCorrRmse_" + filename
    save_path = os.path.join(save_dir, filenameCR)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Correlation maps plot saved to: '{save_path}'")
    return save_path


def ranks(
    predictions,  # Model predictions precipitation (fine predicted)
    targets,  # Ground truth precipitation (fine true)
):
    """
    Compute ranks of predictions compared to targets.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [ensemble_size, batch_size, h, w]
    targets : torch.Tensor or np.array
        Targets of shape [batch_size, h, w]
    Returns
    -------
    np.ndarray(np.float64) of shape [batch_size*h*w,]
    """
    # convert to numpy if tensor :
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    nb_ens, T, L, W = predictions.shape
    predictions_ens = predictions.reshape(nb_ens, T * L * W)
    targets = targets.reshape(1, T * L * W)
    diff = predictions_ens - targets
    mask_leq = (diff <= 0).astype(np.float32)
    mask_l = (diff < 0).astype(np.float32)
    mask = (mask_leq + mask_l) / 2
    return np.sum(mask, axis=0)


def plot_ranks(
    predictions,  # model predictions
    targets,  # ground truth
    variable_names=None,  # list of variable names
    filename="ranks.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Create rank histograms of predictions compared to targets for each variable.

    Parameters
    ----------
    predictions : torch.Tensor or np.array
        Model predictions of shape [ensemble_size, batch_size, num_variables, h, w]
    targets : torch.Tensor or np.array
        Ground truth of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles.
        If None, uses ["VAR_0", "VAR_1", ...]
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots

    Returns
    -------
    save_path : str
        Path to the saved figure
    """
    # Convert tensors → numpy
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()

    ensemble_size, batch_size, num_vars, h, w = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"VAR_{i}" for i in range(num_vars)]
    plot_variable_names = [PlotConfig.get_plot_name(var) for var in variable_names]

    # Figure setup
    fig, axes = plt.subplots(
        1,
        num_vars,
        figsize=(num_vars * figsize_multiplier, figsize_multiplier),
        constrained_layout=True,
    )

    plt.subplots_adjust(
        hspace=0.2, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )
    if num_vars > 1:
        axes = axes.ravel()
    # Handle single subplot case
    else:
        axes = np.array([axes])

    for ax in axes:
        ax.set_box_aspect(1)

    for i, var_name in enumerate(variable_names):
        ax = axes[i]
        plot_name = plot_variable_names[i]

        ranks_predicted = ranks(
            predictions=predictions[:, :, i, :, :],
            targets=targets[:, i, :, :],
        )
        ax.hist(ranks_predicted, bins=np.arange(ensemble_size + 2), density=True)
        ax.plot(
            [0, ensemble_size + 2],
            [1 / (ensemble_size + 2), 1 / (ensemble_size + 2)],
            linestyle="--",
            color="red",
        )

        ax.set_title(plot_name)
        ax.set_xlabel("ranks")
        ax.set_ylabel("frequency")

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ============================================================================
# Plotting Functions Test Suite
# ============================================================================


class TestPlottingFunctions(unittest.TestCase):
    """Unit tests for plotting functions with visible output for styling adjustment."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = "./test_plots"
        os.makedirs(self.output_dir, exist_ok=True)

        # Generate realistic synthetic test data
        np.random.seed(42)
        self.batch_size = 50
        self.num_vars = 4
        self.h = 64
        self.w = 64

        if self.logger:
            self.logger.info(
                f"Test setup complete - output directory: {self.output_dir}"
            )
            self.logger.info(
                f"Batch size: {self.batch_size}, Variables: {self.num_vars}, Resolution: {self.h}x{self.w}"
            )

        # Create correlated data for realistic plots
        x = np.linspace(0, 4 * np.pi, self.w)
        y = np.linspace(0, 4 * np.pi, self.h)
        X, Y = np.meshgrid(x, y)

        patterns = [
            np.sin(X) * np.cos(Y),
            np.exp(-0.1 * (X - 10) ** 2 - 0.1 * (Y - 10) ** 2),
            X * Y / 100,
            np.sin(0.5 * X) * np.cos(0.5 * Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y),
        ]

        self.predictions = np.zeros((self.batch_size, self.num_vars, self.h, self.w))
        self.targets = np.zeros((self.batch_size, self.num_vars, self.h, self.w))
        self.coarse_inputs = np.zeros((self.batch_size, self.num_vars, self.h, self.w))

        for i in range(self.num_vars):
            base_pattern = patterns[i % len(patterns)]
            for b in range(self.batch_size):
                noise_pred = np.random.normal(0, 0.1, (self.h, self.w))
                noise_target = np.random.normal(0, 0.1, (self.h, self.w))
                noise_coarse = np.random.normal(0, 0.2, (self.h, self.w))

                scale = 1.0 + 0.1 * np.random.random()
                offset = 0.1 * np.random.random()

                self.predictions[b, i] = base_pattern * scale + offset + noise_pred
                self.targets[b, i] = (
                    base_pattern * (scale + 0.05) + offset + 0.05 + noise_target
                )
                self.coarse_inputs[b, i] = (
                    base_pattern * (scale - 0.1) + offset - 0.1 + noise_coarse
                )

        self.variable_names = [
            "Temperature (K)",
            # "Pressure (hPa)",
            "VAR_D2M (K)",
            "Humidity (%)",
            "Wind Speed (m/s)",
        ]

        # Create lat/lon arrays for spatial tests
        self.lat = np.linspace(-90, 90, self.h)
        self.lon = np.linspace(-180, 180, self.w)

        # Create comprehensive metrics history
        self.valid_metrics_history = {}
        metrics = ["rmse", "mae", "r2"]

        for var in self.variable_names:
            var_key = var.split(" ")[0]
            for metric in metrics:
                base_val_pred = 0.8 if metric == "r2" else 1.0
                base_val_coarse = 0.6 if metric == "r2" else 1.5
                decay = np.linspace(0, 0.3, 10)

                if metric == "r2":
                    self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"] = (
                        base_val_pred + decay
                    )
                    self.valid_metrics_history[f"{var_key}_coarse_vs_fine_{metric}"] = (
                        base_val_coarse + decay * 0.5
                    )
                else:
                    self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"] = (
                        base_val_pred - decay
                    )
                    self.valid_metrics_history[f"{var_key}_coarse_vs_fine_{metric}"] = (
                        base_val_coarse - decay * 0.5
                    )

        # Add average metrics
        for metric in metrics:
            self.valid_metrics_history[f"average_pred_vs_fine_{metric}"] = (
                0.1 + np.linspace(0, 0.2, 10)
            )
            self.valid_metrics_history[f"average_coarse_vs_fine_{metric}"] = (
                0.7 + np.linspace(0, 0.2, 10)
            )

        # Loss histories
        self.train_loss_history = np.exp(
            -np.linspace(0, 2, 20)
        ) + 0.1 * np.random.random(20)
        self.valid_loss_history = np.exp(
            -np.linspace(0, 1.5, 20)
        ) + 0.15 * np.random.random(20)

    # ============================================================================
    # SINGLE COMPREHENSIVE TEST FOR EACH DIAGNOSTIC METHOD
    # ============================================================================

    def test_validation_hexbin_comprehensive(self):
        """Comprehensive test for validation hexbin plots."""
        if self.logger:
            self.logger.info("Testing validation hexbin plots comprehensively")

        # Test 1: Standard configuration
        expected_path = plot_validation_hexbin(
            predictions=self.predictions,
            targets=self.targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_hexbin_standard.png",
            figsize_multiplier=5,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            filename="comparison_hexbin_standard.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_validation_hexbin(
            predictions=predictions_tensor,
            targets=targets_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_hexbin_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            filename="comparison_hexbin_torch.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        single_pred = self.predictions[:, 0:1, :, :]
        single_target = self.targets[:, 0:1, :, :]
        single_coarse = self.coarse_inputs[:, 0:1, :, :]

        expected_path = plot_validation_hexbin(
            predictions=single_pred,
            targets=single_target,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_hexbin_single.png",
            figsize_multiplier=6,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_comparison_hexbin(
            predictions=single_pred,
            targets=single_target,
            coarse_inputs=single_coarse,
            variable_names=[self.variable_names[0]],
            filename="comparison_hexbin_single.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All validation hexbin tests passed")

    def test_validation_pdfs_comprehensive(self):
        """Comprehensive test for validation PDF plots."""
        if self.logger:
            self.logger.info("Testing validation PDF plots comprehensively")

        # Test 1: Standard configuration with coarse inputs
        expected_path = plot_validation_pdfs(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Without coarse inputs
        expected_path = plot_validation_pdfs(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=None,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_no_coarse.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_validation_pdfs(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_pdfs_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All validation PDF tests passed")

    def test_power_spectra_comprehensive(self):
        """Comprehensive test for power spectra plots."""
        if self.logger:
            self.logger.info("Testing power spectra plots comprehensively")

        dlat = np.abs(self.lat[1] - self.lat[0])
        dlon = np.abs(self.lon[1] - self.lon[0])

        # Test 1: Standard configuration
        expected_path = plot_power_spectra(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Without coarse inputs
        expected_path = plot_power_spectra(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=None,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_no_coarse.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)
        expected_path = plot_power_spectra(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            dlat=dlat,
            dlon=dlon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="power_spectra_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All power spectra tests passed")

    def test_spatiotemporal_histograms_comprehensive(self):
        """Comprehensive test for spatiotemporal histograms."""
        if self.logger:
            self.logger.info("Testing spatiotemporal histograms comprehensively")

        class MockSteps:
            latitude = 180
            longitude = 360

        steps = MockSteps()

        # Test 1: Dense data
        tindex_lim = (0, 365)
        # n_samples = 2000
        centers = []
        tindices = []

        clusters = [
            {
                "lat_range": (30, 60),
                "lon_range": (200, 250),
                "time_range": (0, 100),
                "n": 500,
            },
            {
                "lat_range": (10, 40),
                "lon_range": (100, 150),
                "time_range": (100, 200),
                "n": 400,
            },
            {
                "lat_range": (50, 80),
                "lon_range": (300, 350),
                "time_range": (200, 300),
                "n": 600,
            },
            {
                "lat_range": (0, 30),
                "lon_range": (50, 100),
                "time_range": (300, 365),
                "n": 500,
            },
        ]

        for cluster in clusters:
            for _ in range(cluster["n"]):
                lat = np.random.randint(
                    cluster["lat_range"][0], cluster["lat_range"][1]
                )
                lon = np.random.randint(
                    cluster["lon_range"][0], cluster["lon_range"][1]
                )
                tindex = np.random.randint(
                    cluster["time_range"][0], cluster["time_range"][1]
                )
                centers.append((lat, lon))
                tindices.append(tindex)

        expected_path = plot_spatiotemporal_histograms(
            steps=steps,
            tindex_lim=tindex_lim,
            centers=centers,
            tindices=tindices,
            mode="train",
            filename="spatiotemporal_dense_",
            save_dir=self.output_dir,
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All spatiotemporal histogram tests passed")

    def test_plot_surface_comprehensive(self):
        """Comprehensive test for surface plots."""
        if self.logger:
            self.logger.info("Testing surface plots comprehensively")

        # Test case 1: Standard configuration
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Create synthetic data
        batch_size = 1
        n_vars = 3
        h, w = 48, 68

        # Create spatial patterns
        x = np.linspace(0, 3 * np.pi, w)
        y = np.linspace(0, 3 * np.pi, h)
        X, Y = np.meshgrid(x, y)

        # Initialize arrays
        coarse_inputs = np.zeros((batch_size, n_vars, h, w))
        targets = np.zeros((batch_size, n_vars, h, w))
        pred = np.zeros((batch_size, n_vars, h, w))

        base_patterns = [
            np.sin(X / 2) * np.cos(Y / 2),
            np.exp(-0.01 * (X - 24) ** 2 - 0.01 * (Y - 24) ** 2),
            X * Y / 200,
        ]

        for i in range(n_vars):
            base_pattern = base_patterns[i % len(base_patterns)]
            pattern = base_pattern * 20 + 280  # Temperature-like

            coarse_inputs[0, i] = pattern + np.random.randn(h, w) * 2
            targets[0, i] = pattern + np.random.randn(h, w) * 1
            pred[0, i] = targets[0, i] + np.random.randn(h, w) * 0.3

        variable_names = ["Temperature", "Pressure", "Humidity"]
        timestamp = datetime(2024, 1, 1, 12, 0)

        # Test with numpy arrays
        expected_path = plot_surface(
            coarse_inputs=coarse_inputs,
            targets=targets,
            predictions=pred,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_surface_standard.png",
            save_dir=self.output_dir,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test with PyTorch tensors
        coarse_inputs_tensor = torch.from_numpy(coarse_inputs.copy())
        targets_tensor = torch.from_numpy(targets.copy())
        pred_tensor = torch.from_numpy(pred.copy())

        expected_path = plot_surface(
            coarse_inputs=coarse_inputs_tensor,
            targets=targets_tensor,
            predictions=pred_tensor,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_surface_torch.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All surface plot tests passed")

    def test_plot_global_surface_robinson_comprehensive(self):
        """Comprehensive test for global Robinson surface plots."""
        if self.logger:
            self.logger.info("Testing global Robinson surface plots comprehensively")

        # GLOBAL domain
        H, W = 90, 180
        lat_1d = np.linspace(-90, 90, H)
        lon_1d = np.linspace(-180, 180, W)

        batch_size = 1
        n_vars = 3

        # Create synthetic global patterns
        x = np.linspace(-np.pi, np.pi, W)
        y = np.linspace(-np.pi / 2, np.pi / 2, H)
        X, Y = np.meshgrid(x, y)

        coarse_inputs = np.zeros((batch_size, n_vars, H, W))
        targets = np.zeros((batch_size, n_vars, H, W))
        pred = np.zeros((batch_size, n_vars, H, W))

        base_patterns = [np.sin(X) * np.cos(Y), np.cos(2 * X) * np.sin(Y), X * Y]

        for i in range(n_vars):
            pattern = base_patterns[i] * 10 + 280
            coarse_inputs[0, i] = pattern + np.random.randn(H, W) * 2
            targets[0, i] = pattern + np.random.randn(H, W)
            pred[0, i] = targets[0, i] + np.random.randn(H, W) * 0.3

        variable_names = ["Temperature", "Pressure", "Humidity"]
        timestamp = datetime(2024, 1, 1, 12, 0)

        # Test with numpy arrays
        expected_path = plot_global_surface_robinson(
            predictions=pred,
            targets=targets,
            coarse_inputs=coarse_inputs,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_global_robinson_standard.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))

        # Test with PyTorch tensors
        expected_path = plot_global_surface_robinson(
            predictions=torch.from_numpy(pred),
            targets=torch.from_numpy(targets),
            coarse_inputs=torch.from_numpy(coarse_inputs),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            timestamp=timestamp,
            variable_names=variable_names,
            filename="plot_global_robinson_torch.png",
            save_dir=self.output_dir,
        )

        self.assertTrue(os.path.exists(expected_path))

        if self.logger:
            self.logger.info("✅ All global Robinson surface plot tests passed")

    def test_plot_mae_map_comprehensive(self):
        """Comprehensive test for time-averaged MAE spatial map plots."""
        if self.logger:
            self.logger.info("Testing MAE map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_MAE_map(
            predictions=predictions,
            targets=targets,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mae_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_MAE_map(
            predictions=torch.from_numpy(predictions),
            targets=torch.from_numpy(targets),
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mae_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Single variable
        expected_path = plot_MAE_map(
            predictions=predictions[:, 0:1],
            targets=targets[:, 0:1],
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            variable_names=[self.variable_names[0]],
            save_dir=self.output_dir,
            filename="validation_mae_map_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All MAE map plot tests passed")

    def test_plot_dry_frequency_map_comprehensive(self):
        """Comprehensive test for dry frequency map plots."""
        if self.logger:
            self.logger.info("Testing dry frequency map plots comprehensively")

        # Regional lat/lon
        lat_1d = np.linspace(30, 50, 48)
        lon_1d = np.linspace(-120, -80, 68)

        # Matching spatial resolution
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]

        # Test 1: Standard numpy inputs
        expected_path = plot_dry_frequency_map(
            predictions=predictions[:, 0, :, :],
            targets=targets[:, 0, :, :],
            threshold=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_dry_frequency_map_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: PyTorch tensors
        expected_path = plot_dry_frequency_map(
            predictions=torch.from_numpy(predictions[:, 0, :, :]),
            targets=torch.from_numpy(targets[:, 0, :, :]),
            threshold=1,
            lat_1d=lat_1d,
            lon_1d=lon_1d,
            save_dir=self.output_dir,
            filename="validation_dry_frequency_map_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        if self.logger:
            self.logger.info("✅ All dry frequency map plot tests passed")

    def test_dry_frequency_map(self):
        if self.logger:
            self.logger.info(
                "Testing dry frequency map compute function comprehensively"
            )
        predictions = self.predictions[:, :, :48, :68]
        # Test 1 : standard numpy inputs
        arr = dry_frequency_map(predictions[:, 0, :, :], 1)
        self.assertTrue(arr.shape == predictions.shape[-2:])
        # Test 1 : torch tensors
        arr = dry_frequency_map(torch.from_numpy(predictions[:, 0, :, :]), 1)
        self.assertTrue(arr.shape == predictions.shape[-2:])
        if self.logger:
            self.logger.info("✅ All dry frequency tests passed")

    def test_metric_plots_comprehensive(self):
        """Comprehensive test for metric plots."""
        if self.logger:
            self.logger.info("Testing metric plots comprehensively")

        # Test 1: Metric histories
        expected_path = plot_metric_histories(
            valid_metrics_history=self.valid_metrics_history,
            variable_names=[name.split(" ")[0] for name in self.variable_names],
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="metric_histories_comprehensive",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Loss histories
        expected_path = plot_loss_histories(
            train_loss_history=self.train_loss_history,
            valid_loss_history=self.valid_loss_history,
            save_dir=self.output_dir,
            filename="loss_histories_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 3: Average metrics
        expected_path = plot_average_metrics(
            valid_metrics_history=self.valid_metrics_history,
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="average_metrics_standard.png",
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All metric plot tests passed")

    def test_plot_metrics_heatmap_comprehensive(self):
        """Comprehensive test for validation metrics heatmap."""

        if self.logger:
            self.logger.info("Testing metrics heatmap")

        # Local dummy MetricTracker
        class DummyMetricTracker:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.count = len(self.values)

            def getmean(self):
                return float(np.mean(self.values)) if self.count > 0 else np.nan

        # Fake MetricTracker-based metrics dict
        valid_metrics_trackers = {}

        for var in self.variable_names:
            var_key = var.split(" ")[0]
            for metric in ["rmse", "mae", "r2"]:
                # Reuse the existing synthetic histories
                history = self.valid_metrics_history[f"{var_key}_pred_vs_fine_{metric}"]
                valid_metrics_trackers[f"{var_key}_pred_vs_fine_{metric}"] = (
                    DummyMetricTracker(history)
                )

        expected_path = plot_metrics_heatmap(
            valid_metrics_history=valid_metrics_trackers,
            variable_names=[name.split(" ")[0] for name in self.variable_names],
            metric_names=["rmse", "mae", "r2"],
            save_dir=self.output_dir,
            filename="metrics_heatmap_comprehensive",
        )

        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ Metrics heatmap test passed")

    def test_qq_quantiles_comprehensive(self):
        """Comprehensive test for QQ-quantiles plots."""
        if self.logger:
            self.logger.info("Testing QQ-quantiles plots comprehensively")

        # Test 1: Standard configuration with all parameters
        expected_path = plot_qq_quantiles(
            predictions=self.predictions,
            targets=self.targets,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            save_dir=self.output_dir,
            filename="qq_quantiles_standard.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 4: Single variable (edge case)
        expected_path = plot_qq_quantiles(
            predictions=self.predictions[:, 0:1],  # Keep only first variable
            targets=self.targets[:, 0:1],
            coarse_inputs=self.coarse_inputs[:, 0:1],
            variable_names=["Temperature (K)"],
            quantiles=[0.90, 0.95, 0.99],
            save_dir=self.output_dir,
            filename="qq_quantiles_single_var.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 5: PyTorch tensors
        predictions_tensor = torch.from_numpy(self.predictions)
        targets_tensor = torch.from_numpy(self.targets)
        coarse_tensor = torch.from_numpy(self.coarse_inputs)

        expected_path = plot_qq_quantiles(
            predictions=predictions_tensor,
            targets=targets_tensor,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            quantiles=[0.90, 0.95, 0.975, 0.99, 0.995],
            save_dir=self.output_dir,
            filename="qq_quantiles_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All QQ-quantiles tests passed")

    def test_mv_correlation_comprehensive(self):
        """Test for temporal correlation between pairs of variables."""

        # Define lat lon grid
        w = self.predictions.shape[2]
        h = self.predictions.shape[3]
        dlat = 20
        dlon = 40
        lat1 = 30
        lon1 = -120
        lon, lat = np.meshgrid(
            np.linspace(lon1, lon1 + dlon, w), np.linspace(lat1, lat1 + dlat, h)
        )

        # Test 1: Standard configuration Numpy arrays
        expected_path = plot_validation_mvcorr(
            predictions=self.predictions,
            targets=self.targets,
            lat=lat,
            lon=lon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mvcorr_numpy.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_validation_mvcorr(
            predictions=self.predictions,
            targets=self.targets,
            lat=lat,
            lon=lon,
            coarse_inputs=self.coarse_inputs,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mvcorr_numpy.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        # Test 2: Standard configuration PyTorch tensors
        coarse_tensor = torch.from_numpy(self.coarse_inputs.copy())
        fine_tensor = torch.from_numpy(self.targets.copy())
        pred_tensor = torch.from_numpy(self.predictions.copy())
        expected_path = plot_validation_mvcorr(
            predictions=pred_tensor,
            targets=fine_tensor,
            lat=lat,
            lon=lon,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="validation_mvcorr_torch.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        expected_path = plot_validation_mvcorr(
            predictions=pred_tensor,
            targets=fine_tensor,
            lat=lat,
            lon=lon,
            coarse_inputs=coarse_tensor,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="comparison_mvcorr_torch.png",
            figsize_multiplier=3,
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All mv correlation tests passed")

    def test_ranks(self):
        if self.logger:
            self.logger.info("Testing ranks compute function comprehensively")
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, 0, :48, :68]
        ensemble_size = 10
        # Test 1 : standard numpy inputs
        predictions_repeated = np.repeat(
            predictions[None, :, 0, :, :], axis=0, repeats=ensemble_size
        )
        arr = ranks(predictions=predictions_repeated, targets=targets)
        self.assertTrue(arr.shape == targets.flatten().shape)
        # Test 2 : torch tensors
        predictions_repeated_torch = torch.from_numpy(predictions_repeated)
        targets_torch = torch.from_numpy(targets)
        arr_torch = ranks(predictions=predictions_repeated_torch, targets=targets_torch)
        self.assertTrue(arr_torch.shape == targets.flatten().shape)

        if self.logger:
            self.logger.info("✅ All ranks tests passed")

    def test_plot_ranks(self):
        if self.logger:
            self.logger.info("Testing plot_ranks function comprehensively")
        predictions = self.predictions[:, :, :48, :68]
        targets = self.targets[:, :, :48, :68]
        ensemble_size = 10
        # Test 1 : standard numpy inputs
        predictions_repeated = np.repeat(
            predictions[None, :, :, :, :], axis=0, repeats=ensemble_size
        )
        expected_path = plot_ranks(
            predictions=predictions_repeated,
            targets=targets,
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="ranks.png",
        )
        self.assertTrue(
            os.path.exists(expected_path), f"File not found: {expected_path}"
        )
        # Test 2 : torch inputs
        expected_path_torch = plot_ranks(
            predictions=torch.from_numpy(predictions_repeated),
            targets=torch.from_numpy(targets),
            variable_names=self.variable_names,
            save_dir=self.output_dir,
            filename="ranks_torch.png",
        )
        self.assertTrue(
            os.path.exists(expected_path_torch), f"File not found: {expected_path}"
        )

        if self.logger:
            self.logger.info("✅ All ranks plot tests passed")

    def tearDown(self):
        """Clean up after tests."""
        # Note: We don't remove the output directory so you can inspect the plots
        if self.logger:
            self.logger.info(
                f"Plotting tests completed - plots available in: {self.output_dir}"
            )


# ----------------------------------------------------------------------------
