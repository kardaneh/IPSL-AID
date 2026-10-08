# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os

os.environ.setdefault(
    "CARTOPY_DATA_DIR",
    "/leonardo_work/EUHPC_D36_053/cartopy_data",
)

import mpltex
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature


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
        "error": "Reds",
        "divergence": "seismic",
        "curl": "seismic",
        "ssr": "seismic",
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


def to_numpy_4d(data):
    """
    Convert input data to a four-dimensional NumPy array.

    The returned array follows the dimension order
    "[time, variable, latitude, longitude]".

    Parameters
    ----------
    data : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Input data to convert.

    Returns
    -------
    numpy.ndarray
        Four-dimensional array with shape
        "[time, variable, latitude, longitude]".

    Raises
    ------
    ValueError
        If the converted data does not have four dimensions.
    """

    # torch.Tensor -> numpy.ndarray
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()

    # xarray.DataArray -> numpy.ndarray
    elif hasattr(data, "dims") and hasattr(data, "values"):
        data = data.transpose("time", "latitude", "longitude").values
        data = data[:, None, :, :]

    # list of xarray.DataArray -> numpy.ndarray
    elif isinstance(data, list):
        arrays = []

        for da in data:
            if hasattr(da, "dims") and hasattr(da, "values"):
                arr = da.transpose("time", "latitude", "longitude").values
            else:
                arr = np.asarray(da)

            arrays.append(arr)

        data = np.stack(arrays, axis=1)

    # numpy-like input
    else:
        data = np.asarray(data)

    if data.ndim != 4:
        raise ValueError(
            f"Expected data with 4 dimensions [time, variable, latitude, longitude], "
            f"got shape {data.shape}"
        )

    return data


def plot_validation_pdfs(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    variable_names=None,  # List of variable names
    filename="validation_pdfs.png",
    save_dir="./results",
    figsize_multiplier=4,  # Base size per subplot
    save_npz=False,
):
    """
    Create PDF (Probability Density Function) plots comparing distributions of
    model predictions vs ground truth for all variables.

    Parameters
    ----------
    predictions : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray, optional
        Coarse inputs of shape [batch_size, num_variables, h, w]
    variable_names : list of str, optional
        Names of the variables for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots
    save_npz : bool, optional
        If True, saves the PDF diagnostics to a compressed .npz file.

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
    # Convert tensors, NumPy arrays, or xarray objects to NumPy 4D arrays
    predictions = to_numpy_4d(predictions)
    targets = to_numpy_4d(targets)

    if coarse_inputs is not None:
        coarse_inputs = to_numpy_4d(coarse_inputs)

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

    if save_npz:
        pdf_npz_data = {}

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

        if save_npz:
            key = f"{var_name}__pdf__"

            pdf_npz_data[key + "bin_centers"] = bin_centers
            pdf_npz_data[key + "log_pred"] = log_hist_pred
            pdf_npz_data[key + "log_truth"] = log_hist_target

            pdf_npz_data[key + "mean_pred"] = pred_mean
            pdf_npz_data[key + "std_pred"] = pred_std
            pdf_npz_data[key + "mean_truth"] = target_mean
            pdf_npz_data[key + "std_truth"] = target_std
            pdf_npz_data[key + "kl"] = kl_divergence
            pdf_npz_data[key + "corr"] = correlation

            if coarse_inputs is not None:
                pdf_npz_data[key + "log_coarse"] = log_hist_coarse
                pdf_npz_data[key + "mean_coarse"] = coarse_mean
                pdf_npz_data[key + "std_coarse"] = coarse_std

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if save_npz:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(npz_path, **pdf_npz_data)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
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
    save_npz=False,
):
    """
    Calculate and plot power spectra with proper physical wavenumbers.

    Parameters
    ----------
    predictions : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Model predictions of shape [batch_size, num_variables, nh, nw]
    targets : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Ground truth of shape [batch_size, num_variables, nh, nw]
    dlat : float
        Grid spacing in latitude (degrees)
    dlon : float
        Grid spacing in longitude (degrees)
    coarse_inputs : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray, optional
        Coarse inputs of shape [batch_size, num_variables, nh, nw]
    variable_names : list of str, optional
        Names of the variable names for subplot titles
    filename : str, optional
        Output filename
    save_dir : str, optional
        Directory to save the plot
    figsize_multiplier : int, optional
        Base size multiplier for subplots
    save_npz : bool, optional
        If True, saves the PDF diagnostics to a compressed .npz file.

    Returns
    -------
    None
    """
    # Convert tensors, NumPy arrays, or xarray objects to NumPy 4D arrays
    predictions = to_numpy_4d(predictions)
    targets = to_numpy_4d(targets)

    if coarse_inputs is not None:
        coarse_inputs = to_numpy_4d(coarse_inputs)

    batch_size, num_vars, nh, nw = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"Variable {i + 1}" for i in range(num_vars)]

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

    if save_npz:
        spectra_npz_data = {}

        spectra_npz_data["__meta__dlat"] = dlat
        spectra_npz_data["__meta__dlon"] = dlon
        spectra_npz_data["__meta__variables"] = np.array(variable_names)

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

        if save_npz:
            key = f"{var_name}__spectra__"

            spectra_npz_data[key + "k"] = k_centers
            spectra_npz_data[key + "psd_pred"] = psd1d_pred
            spectra_npz_data[key + "psd_truth"] = psd1d_target

            if coarse_inputs is not None:
                spectra_npz_data[key + "psd_coarse"] = psd1d_coarse

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

    if save_npz:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(npz_path, **spectra_npz_data)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
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
    save_npz=False,
):
    """
    Create QQ-plats at different quantiles comparing model predictions and
    coarse inputs against ground truth.

    For each variable, plots quantiles of predictions and coarse inputs
    against quantiles of ground truth with a 1:1 reference line.

    Parameters
    ----------
    predictions : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Model predictions of shape [batch_size, num_variables, h, w]
    targets : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
        Ground truth of shape [batch_size, num_variables, h, w]
    coarse_inputs : torch.Tensor, numpy.ndarray, xarray.DataArray, or list of xarray.DataArray
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
    save_npz : bool, optional
        If True, saves the PDF diagnostics to a compressed .npz file.

    Returns
    -------
    save_path : str
        Path to the saved figure
    """

    # Convert tensors, NumPy arrays, or xarray objects to NumPy 4D arrays
    predictions = to_numpy_4d(predictions)
    targets = to_numpy_4d(targets)
    coarse_inputs = to_numpy_4d(coarse_inputs)

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
    )

    if num_vars > 1:
        axes = axes.ravel()
    # Handle single subplot case
    else:
        axes = np.array([axes])

    for ax in axes:
        ax.set_box_aspect(1)

    plt.subplots_adjust(
        hspace=0.1,
        wspace=0.3,
        left=0.1,
        right=0.9,
        top=0.9,
        bottom=0.1,
    )

    if save_npz:
        qq_npz_data = {}

        qq_npz_data["__meta__variables"] = np.array(variable_names)
        qq_npz_data["__meta__quantiles"] = np.array(quantiles)

    for i, var_name in enumerate(variable_names):
        linestyles = mpltex.linestyle_generator(lines=[])
        ax = axes[i]
        plot_name = plot_variable_names[i]

        pred_vals = PlotConfig.convert_units(var_name, predictions[:, i])
        target_vals = PlotConfig.convert_units(var_name, targets[:, i])
        coarse_vals = PlotConfig.convert_units(var_name, coarse_inputs[:, i])

        # Compute quantiles
        qs_target = np.quantile(target_vals, quantiles)
        qs_pred = np.quantile(pred_vals, quantiles)
        qs_coarse = np.quantile(coarse_vals, quantiles)

        if save_npz:
            key = f"{var_name}__qq__"

            qq_npz_data[key + "quantiles"] = np.array(quantiles)
            qq_npz_data[key + "truth"] = qs_target
            qq_npz_data[key + "pred"] = qs_pred
            qq_npz_data[key + "coarse"] = qs_coarse

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

    if save_npz:
        npz_path = os.path.splitext(save_path)[0] + ".npz"
        np.savez_compressed(npz_path, **qq_npz_data)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metrics_heatmap(
    summaries,
    metric_names,
    filename="validation_metrics_heatmap.png",
    save_dir="./results",
    figsize_multiplier=4,
):
    """
    Plot a heatmap of summary metrics.

    Parameters
    ----------
    summaries : list of dict
        Summary metrics returned by summary_metrics.
    metric_names : list of str
        Metrics to display.
    filename : str, optional
        Output filename.
    save_dir : str, optional
        Directory where the image is saved.
    figsize_multiplier : float, optional
        Controls overall figure size.

    Returns
    -------
    str
        Path to the saved figure.
    """
    dataset_names = [row["dataset"] for row in summaries]

    values = np.asarray(
        [[row.get(metric, np.nan) for metric in metric_names] for row in summaries],
        dtype=np.float64,
    )

    fig_width = figsize_multiplier + len(metric_names)
    fig_height = 0.6 * len(dataset_names) + figsize_multiplier / 2

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
    )

    sns.heatmap(
        values,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        linewidths=0.8,
        cbar=True,
        xticklabels=metric_names,
        yticklabels=dataset_names,
    )

    ax.set_title("Validation metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Dataset")

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        filename,
    )

    plt.savefig(
        save_path,
        bbox_inches="tight",
    )

    plt.close(fig)

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

    # Convert tensors, NumPy arrays, or xarray objects to NumPy 4D arrays
    predictions = to_numpy_4d(predictions)
    targets = to_numpy_4d(targets)
    coarse_inputs = to_numpy_4d(coarse_inputs)

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
        squeeze=False,
    )

    # Main title
    if timestamp is not None:
        #    fig.suptitle(
        #        f"Forecast for {timestamp.strftime('%Y-%m-%d %H:%M')}",
        #        fontsize=16, y=1.02
        #    )
        print(f"Forecast for {timestamp.strftime('%Y-%m-%d %H:%M')}")

    # Plot each variable
    for col_idx in range(n_vars):
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
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

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
