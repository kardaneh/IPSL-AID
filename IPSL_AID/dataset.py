# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import tempfile
import torch
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from IPSL_AID.utils import EasyDict
from torch.utils.data import Dataset
import torchvision
import glob
import os
import json
import shutil
import unittest


def stats(ds, logger, input_dir, norm_mapping=dict()):
    """
    Compute statistics and histograms for variables across a NetCDF dataset.

    This function analyzes a NetCDF dataset to compute descriptive statistics
    for both coordinates and data variables, and generates histograms for
    visualization. It also applies predefined normalization constants for
    specific weather variables.

    Parameters
    ----------
    ds : xarray.Dataset
        NetCDF dataset to process.
    logger : logging.Logger
        Logger instance for logging messages and statistics.
    input_dir : str
        Directory containing a statistics.json file with precomputed
        normalization statistics.
    norm_mapping : dict, optional
        Dictionary to store computed statistics. If empty, will be populated.
        Default is empty dict.

    Returns
    -------
    norm_mapping : dict
        Dictionary mapping variable names to their computed statistics.
        For coordinates: min, max, mean, std.
        For data variables: min, max, mean, std, q1, q3, iqr, median.
    steps : EasyDict
        Dictionary containing coordinate step sizes and lengths.

    Notes
    -----
    - Computes statistics for all coordinates (excluding datetime) and data variables.
    - Generates histograms for data variables and saves them as PNG files.
    - Applies predefined normalization constants for specific weather variables.
    - Saves statistics to a JSON file in the output directory.
    - Handles multi-dimensional coordinates with warnings.
    - Uses EasyDict for convenient attribute access to nested dictionaries.
    """

    # stats_file_path = os.path.join(output_dir, "statistics.json")

    logger.info("Starting statistics computation...")
    steps = EasyDict()

    stats_loaded = False
    # Load stats from JSON if available
    if input_dir is not None:
        stats_path = os.path.join(input_dir, "statistics.json")
        if os.path.isfile(stats_path):
            logger.info(f"Loading normalization statistics from {stats_path}")

            with open(stats_path, "r") as f:
                raw_stats = json.load(f)

            # Load all statistics from JSON into norm_mapping
            for key, values in raw_stats.items():
                norm_mapping[key] = EasyDict(values)

            stats_loaded = True

    # Use manual stats
    if not stats_loaded:
        logger.info("No statistics.json found, using manual constants")

        RAW_CONSTANTS = {
            "VAR_2T": {"mean": 2.8504e02, "std": 12.7438},
            "VAR_10U": {"mean": 4.4536e-01, "std": 3.4649},
            "VAR_10V": {"mean": -1.1892e-01, "std": 3.7420},
            "VAR_TP": {"mean": 9.4189e-05, "std": 2.6393e-04},
            "VAR_D2M": {"mean": 2.8250e02, "std": 5.5930},
            "VAR_SSTK": {"mean": 2.92517e02, "std": 8.79515},
            "VAR_SKT": {"mean": 2.86780e02, "std": 13.4919},
            "VAR_ST": {"mean": 2.86905e02, "std": 13.4027},
            "VAR_TCWV": {"mean": 1.97549e01, "std": 13.4064},
        }

        RESID_CONSTANTS = {
            "VAR_2T_residual": {"mean": -9.4627e-05, "std": 1.6042},
            "VAR_10U_residual": {"mean": -1.3833e-03, "std": 1.0221},
            "VAR_10V_residual": {"mean": -1.5548e-03, "std": 1.0384},
            "VAR_TP_residual": {"mean": -4.0417e-08, "std": 2.8678e-04},
            "VAR_D2M_residual": {"mean": 4.6380e-01, "std": 1.0602},
            "VAR_SSTK_residual": {"mean": -3.50143e-02, "std": 7.58565e-01},
            "VAR_SKT_residual": {"mean": 3.11868e-02, "std": 2.04542},
            "VAR_ST_residual": {"mean": 3.21775e-02, "std": 2.27347},
            "VAR_TCWV_residual": {"mean": 3.90753e-02, "std": 1.73849},
        }

        for var_name in ds.data_vars:
            var_name_residual = f"{var_name}_residual"
            var_name_coarse = f"{var_name}_coarse"

            norm_mapping[var_name_residual] = EasyDict()
            norm_mapping[var_name_coarse] = EasyDict()

            if var_name in RAW_CONSTANTS:
                norm_mapping[var_name_coarse].vmean = RAW_CONSTANTS[var_name]["mean"]
                norm_mapping[var_name_coarse].vstd = RAW_CONSTANTS[var_name]["std"]

            if var_name_residual in RESID_CONSTANTS:
                norm_mapping[var_name_residual].vmean = RESID_CONSTANTS[
                    var_name_residual
                ]["mean"]
                norm_mapping[var_name_residual].vstd = RESID_CONSTANTS[
                    var_name_residual
                ]["std"]

    logger.info("------------------------------------------")

    for cname in ds.coords:
        cdata = ds[cname].values
        steps[cname] = len(cdata)
        steps[f"d_{cname}"] = abs(cdata[1] - cdata[0])
        # Skip multi-dimensional coordinates (rare but possible)
        if cdata.ndim != 1:
            logger.warning(f"[COORD] Skipping '{cname}' because it is not 1-D")
            continue

        # Skip datetime unless desired
        if np.issubdtype(cdata.dtype, np.datetime64):
            logger.info(f"[COORD] Skipping time coordinate '{cname}' (datetime64)")
            continue

        vmin = float(np.min(cdata))
        vmax = float(np.max(cdata))
        vmean = float(np.mean(cdata))
        vstd = float(np.std(cdata))
        # q1 = float(np.percentile(cdata, 25))
        # q3 = float(np.percentile(cdata, 75))
        # iqr = q3 - q1 if q3 != q1 else 1.0
        # median = float(np.median(cdata))

        norm_mapping[cname] = EasyDict()
        norm_mapping[cname].vmin = vmin
        norm_mapping[cname].vmax = vmax
        norm_mapping[cname].vmean = vmean
        norm_mapping[cname].vstd = vstd

    logger.info("------ Coordinate / Dimension Sizes ------")
    for key, value in steps.items():
        logger.info(f" └── {key}: {value}")
    logger.info("------------------------------------------")

    return norm_mapping, steps


def coarse_down_up(fine_filtered, fine_batch, input_shape=(16, 32), axis=0):
    """
    Downscale and then upscale fine-resolution data to compute coarse approximation.

    This function performs a downscaling-upscaling operation to create a coarse
    resolution approximation of fine data. This is commonly used in
    multi-scale analysis, image processing, and super-resolution tasks.

    Parameters
    ----------
    fine_filtered : torch.Tensor or np.ndarray
        Fine-resolution filtered data. Can be of shape (C, Hf, Wf) for multi-channel
        data or (Hf, Wf) for single-channel data. Where C is number of channels,
        Hf is fine height, and Wf is fine width.
    fine_batch : torch.Tensor or np.ndarray
        Fine-resolution target data. Must have same spatial dimensions as
        `fine_filtered`. Shape: (C, Hf, Wf) or (Hf, Wf).
    input_shape : tuple of int, optional
        Target shape (Hc, Wc) for the coarse-resolution data after downscaling.
        Default is (16, 32).
    axis : int, optional
        Axis along which to insert batch dimension if the input lacks one.
        Default is 0.

    Returns
    -------
    coarse_up : torch.Tensor
        Upscaled coarse approximation of the fine data. Same shape as input
        `fine_filtered` without batch dimension.

    Notes
    -----
    - The function ensures that the input tensors have a batch dimension by
      adding one if missing.
    - Uses bilinear interpolation for both downscaling and upscaling operations.
    - The antialias parameter is set to True for better quality resampling.
    - Useful for creating multi-scale representations in image processing and
      computer vision tasks.
    """
    # Ensure batch dimension
    if isinstance(fine_filtered, np.ndarray):
        fine_filtered = torch.from_numpy(fine_filtered)
    if isinstance(fine_batch, np.ndarray):
        fine_batch = torch.from_numpy(fine_batch)

    if fine_filtered.dim() == 3:
        fine_filtered = fine_filtered.unsqueeze(axis)  # (1, C, H, W)
    if fine_batch.dim() == 3:
        fine_batch = fine_batch.unsqueeze(axis)

    # Downscale to coarse resolution
    coarsen_transform = torchvision.transforms.Resize(
        input_shape,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    out_shape = (fine_filtered.shape[-2], fine_filtered.shape[-1])
    interp_transform = torchvision.transforms.Resize(
        out_shape,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )

    coarse_up = interp_transform(coarsen_transform(fine_filtered))
    # Remove batch dimension
    coarse_up = coarse_up.squeeze(0)
    # fine_squeezed = fine_batch.squeeze(0)

    # residual = fine_squeezed - coarse_up

    return coarse_up


def gaussian_filter(
    image, dW, dH, cutoff_W_phys, cutoff_H_phys, epsilon=0.01, margin=8
):
    """
    Apply a Gaussian low-pass filter with controlled attenuation at the cutoff frequency.

    This function performs a 2D Fourier transform of the input field, applies a Gaussian
    weighting in the frequency domain, and inversely transforms it back to the spatial
    domain. Unlike the standard Gaussian filter, this version defines the Gaussian width
    such that the response amplitude reaches a specified attenuation factor (epsilon)
    at the cutoff frequency. Padding with reflection is used to minimize edge artifacts.

    Parameters
    ----------
    image : np.array of shape (H, W)
        Input 2D field to be filtered (temperature, wind component).
    dW : float
        Grid spacing in degrees of longitude.
    dH : float
        Grid spacing in degrees of latitude.
    cutoff_W_phys : float
        Longitudinal cutoff frequency in cycles per degree. Frequencies higher than
        this threshold are attenuated according to the Gaussian response.
    cutoff_H_phys : float
        Latitudinal cutoff frequency in cycles per degree. Frequencies higher than
        this threshold are attenuated according to the Gaussian response.
    epsilon : float, optional
        Desired amplitude response at the cutoff frequency (default: 0.01).
        Lower values produce sharper attenuation and stronger filtering.
    margin : int, optional
        Number of pixels to pad on each side using reflection (default: 8).
        This reduces edge effects in the Fourier transform.

    Returns
    -------
    filtered : ndarray of shape (H, W)
        Real-valued filtered field after inverse Fourier transform and margin cropping.

    Notes
    -----
    - The Gaussian width parameters are computed such that:
      exp(-0.5 * (f_cutoff / sigma)^2) = epsilon,
      leading to sigma = f_cutoff / sqrt(-2 * log(epsilon)).
    - Padding the input with reflective boundaries minimizes spectral leakage
      and discontinuities at image edges.
    - The output field is cropped back to its original size after filtering.
    - This formulation provides more explicit control over filter sharpness
      than the standard Gaussian low-pass implementation.
    """
    # H = number of latitude grid points, W = number of longitude grid points
    H, W = image.shape

    # Add reflective padding around the input image to reduce FFT edge artifacts
    img_pad = np.pad(
        image, margin, mode="reflect"
    )  # margin defines how many pixels to pad on each side

    # 2D Fast Fourier Transform (FFT) of the padded image
    fft = np.fft.fft2(img_pad)

    fH = np.fft.fftfreq(img_pad.shape[1], d=dH)  # cycles per degree latitude
    fW = np.fft.fftfreq(img_pad.shape[0], d=dW)  # cycles per degree longitude

    # 2D meshgrid of longitude (fW) and latitude (fH) frequencies
    FW, FH = np.meshgrid(fW, fH, indexing="ij")  # shape (H, W)

    assert FW.shape == fft.shape and FH.shape == fft.shape, (
        f"Frequency grid shape mismatch: "
        f"FW.shape={FW.shape}, FH.shape={FH.shape}, but fft.shape={fft.shape}. "
        "Ensure meshgrid construction matches the 2D FFT dimensions."
    )

    # Sigma (-0.5 * (f_cutoff / sigma)^2 ) = epsilon
    sigma_W = cutoff_W_phys / np.sqrt(-2 * np.log(epsilon))
    sigma_H = cutoff_H_phys / np.sqrt(-2 * np.log(epsilon))

    H_filter = np.exp(-0.5 * ((FW / sigma_W) ** 2 + (FH / sigma_H) ** 2))

    fft_filtered = fft * H_filter
    img_filt_pad = np.fft.ifft2(fft_filtered).real
    # The padded margins are discarded to ensure the output has the same dimensions as the input image
    filtered = img_filt_pad[margin:-margin, margin:-margin]

    return filtered


class DataPreprocessor(Dataset):
    """
    Dataset class for preprocessing weather and climate data for machine learning.

    This class handles loading, preprocessing, and sampling of multi-year NetCDF
    weather data with support for multi-scale processing, normalization, and
    spatial-temporal sampling strategies.

    Parameters
    ----------
    years : list of int
        Years of data to include.
    loaded_dfs : xarray.Dataset
        Pre-loaded dataset containing the weather variables.
    constants_file_path : str
        Path to NetCDF file containing constant variables (e.g., topography).
    varnames_list : list of str
        List of variable names to extract from the dataset.
    units_list : list of str
        Units for each variable in varnames_list.
    in_shape : tuple of int, optional
        Target shape (height, width) for coarse resolution. Default is (16, 32).
    batch_size_lat : int, optional
        Height of spatial batch in grid points. Default is 144.
    batch_size_lon : int, optional
        Width of spatial batch in grid points. Default is 144.
    steps : EasyDict, optional
        Dictionary containing grid dimension information. Should include:
        - latitude/lat: number of latitude points
        - longitude/lon: number of longitude points
        - time: number of time steps
        - d_latitude/d_lat: latitude spacing
        - d_longitude/d_lon: longitude spacing
    tbatch : int, optional
        Number of time batches to sample. Default is 1.
    sbatch : int, optional
        Number of spatial batches to sample. Default is 8.
    debug : bool, optional
        Enable debug logging. Default is True.
    mode : str, optional
        Operation mode: "train" or "validation". Default is "train".
    run_type : str, optional
        Run type: "train", "validation", or "inference". Default is "train".
    dynamic_covariates : list of str, optional
        List of dynamic covariate variable names. Default is None.
    dynamic_covariates_dir : str, optional
        Directory containing dynamic covariate files. Default is None.
    time_normalization : str, optional
        Method for time normalization: "linear" or "cos_sin". Default is "linear".
    norm_mapping : dict, optional
        Dictionary containing normalization statistics for variables.
    index_mapping : dict, optional
        Dictionary mapping variable names to indices in the data array.
    normalization_type : dict, optional
        Dictionary specifying normalization type per variable.
    constant_variables : list of str, optional
        List of constant variable names to load. Default is None.
    epsilon : float, optional
        Small value for numerical stability in filtering. Default is 0.02.
    margin : int, optional
        Margin for filtering operations. Default is 8.
    dtype : tuple, optional
        Data types for torch and numpy (torch_dtype, np_dtype).
        Default is (torch.float32, np.float32).
    apply_filter : bool, optional
        Whether to apply Gaussian filtering for multi-scale processing.
        Default is False.
    logger : logging.Logger, optional
        Logger instance for logging messages. Default is None.

    Attributes
    ----------
    const_vars : np.ndarray or None
        Array of constant variables with shape (n_constants, H, W).
    time : xarray.DataArray
        Time coordinate from dataset.
    year : xarray.DataArray
        Year component of time.
    month : xarray.DataArray
        Month component of time.
    day : xarray.DataArray
        Day component of time.
    hour : xarray.DataArray
        Hour component of time.
    year_norm : torch.Tensor
        Normalized year values.
    doy_norm : torch.Tensor or None
        Normalized day-of-year values (linear mode).
    hour_norm : torch.Tensor or None
        Normalized hour values (linear mode).
    doy_sin, doy_cos : torch.Tensor or None
        Sine and cosine of day-of-year (cos_sin mode).
    hour_sin, hour_cos : torch.Tensor or None
        Sine and cosine of hour (cos_sin mode).
    time_batchs : np.ndarray
        Array of time indices for current epoch.
    eval_slices : list of tuple or None
        List of spatial slices for evaluation mode.
    random_centers : list of tuple or None
        List of random spatial centers for training mode.
    center_tracker : list
        Tracks spatial centers for debugging.
    tindex_tracker : list
        Tracks temporal indices for debugging.

    Methods
    -------
    new_epoch()
        Reset time batches and random centers for new training epoch.
    sample_time_steps_by_doy()
        Sample time steps based on day-of-year (DOY) for multi-year continuity.
    sample_random_time_indices()
        Randomly sample time indices for training.
    load_dynamic_covariates()
        Load dynamic covariate data (not fully implemented).
    generate_random_batch_centers(n_batches)
        Generate random spatial centers for batch sampling.
    generate_evaluation_slices()
        Generate deterministic spatial slices for evaluation.
    extract_batch(data, ilat, ilon)
        Extract spatial batch centered at (ilat, ilon) with cyclic longitude.
    filter_batch(fine_patch, fine_block)
        Apply Gaussian low-pass filtering for multi-scale processing.
    normalize(data, stats, norm_type, var_name=None, data_type=None)
        Normalize data using specified statistics and method.
    normalize_time(tindex)
        Return normalized time features for given time index.
    __len__()
        Return total number of samples.
    __getitem__(index)
        Get a single sample with appropriate spatial-temporal sampling.

    Notes
    -----
    - Supports both random (training) and deterministic (validation) sampling.
    - Handles cyclic longitude wrapping for global datasets.
    - Provides multi-scale processing through downscaling/upscaling.
    - Includes time normalization with linear or trigonometric encoding.
    - Can incorporate constant variables (e.g., topography, land-sea mask).
    """

    def __init__(
        self,
        years,
        loaded_dfs,
        constants_file_path,
        varnames_list,
        units_list,
        in_shape=(16, 32),
        batch_size_lat=144,
        batch_size_lon=144,
        steps=dict(),
        tbatch=1,
        sbatch=8,
        debug=True,
        mode="train",
        run_type="train",
        dynamic_covariates=None,
        dynamic_covariates_dir=None,
        time_normalization="linear",
        norm_mapping=None,
        index_mapping=None,
        normalization_type=None,
        constant_variables=None,
        epsilon=0.02,
        margin=8,
        dtype=(torch.float32, np.float32),
        apply_filter=False,
        region_center=None,  # (lat_value, lon_value)
        logger=None,
    ):
        """
        Initialize the DataPreprocessor.

        Parameters
        ----------
        years : list of int
            Years of data to include.
        loaded_dfs : xarray.Dataset
            Pre-loaded dataset containing the weather variables.
        constants_file_path : str
            Path to NetCDF file containing constant variables.
        varnames_list : list of str
            List of variable names to extract.
        units_list : list of str
            Units for each variable.
        in_shape : tuple of int, optional
            Target shape for coarse resolution.
        batch_size_lat : int, optional
            Height of spatial batch.
        batch_size_lon : int, optional
            Width of spatial batch.
        steps : EasyDict, optional
            Grid dimension information.
        tbatch : int, optional
            Number of time batches.
        sbatch : int, optional
            Number of spatial batches.
        debug : bool, optional
            Enable debug logging.
        mode : str, optional
            Operation mode.
        run_type : str, optional
            Run type.
        dynamic_covariates : list of str, optional
            Dynamic covariate variable names.
        dynamic_covariates_dir : str, optional
            Directory for dynamic covariates.
        time_normalization : str, optional
            Time normalization method.
        norm_mapping : dict, optional
            Normalization statistics.
        index_mapping : dict, optional
            Variable to index mapping.
        normalization_type : dict, optional
            Normalization type per variable.
        constant_variables : list of str, optional
            Constant variable names.
        epsilon : float, optional
            Numerical stability value.
        margin : int, optional
            Filter margin.
        dtype : tuple, optional
            Data types for torch and numpy.
        apply_filter : bool, optional
            Apply Gaussian filtering.
        region_center : tuple of float or None
            Fixed geographic center (lat, lon) for spatial sampling.
        logger : logging.Logger, optional
            Logger instance.
        """
        self.constants_file_path = constants_file_path
        self.constant_variables = constant_variables
        self.years = years
        self.varnames_list = varnames_list
        self.units_list = units_list
        self.in_shape = in_shape
        self.batch_size_lat = batch_size_lat
        self.batch_size_lon = batch_size_lon

        if hasattr(steps, "latitude"):
            steps.latitude = steps.latitude
            steps.d_latitude = steps.d_latitude
        elif hasattr(steps, "lat"):
            steps.latitude = steps.lat
            steps.d_latitude = steps.d_lat
        else:
            assert False, (
                f"Missing required latitude coordinate ('latitude' or 'lat'). "
                f"Available keys: {list(vars(steps).keys())}"
            )

        if hasattr(steps, "longitude"):
            steps.longitude = steps.longitude
            steps.d_longitude = steps.d_longitude
        elif hasattr(steps, "lon"):
            steps.longitude = steps.lon
            steps.d_longitude = steps.d_lon
        else:
            assert False, (
                f"Missing required longitude coordinate ('longitude' or 'lon'). "
                f"Available keys: {list(vars(steps).keys())}"
            )

        assert hasattr(steps, "time"), (
            f"Missing required 'time' coordinate. "
            f"Available keys: {list(vars(steps).keys())}"
        )

        self.H = steps.latitude
        self.dH = steps.d_latitude
        self.W = steps.longitude
        self.dW = steps.d_longitude

        self.region_center = region_center

        self.tbatch = tbatch
        self.sbatch = sbatch
        self.stime = 0
        self.debug = debug
        self.mode = mode
        self.run_type = run_type
        self.dynamic_covariates = dynamic_covariates
        self.dynamic_covariates_dir = dynamic_covariates_dir
        self.time_normalization = time_normalization
        self.norm_mapping = norm_mapping
        self.index_mapping = index_mapping
        self.normalization_type = normalization_type
        self.epsilon = epsilon
        self.margin = margin
        self.torch_dtype = dtype[0]
        self.np_dtype = dtype[1]
        self.apply_filter = apply_filter
        self.logger = logger

        if self.apply_filter:
            self.logger.info(f"Fine filtering enabled: {self.apply_filter}")

        # Validate batch sizes
        if self.batch_size_lat > self.H:
            raise ValueError(
                f"batch height {self.batch_size_lat} exceeds latitude dimension {self.H}"
            )
        if self.batch_size_lon > self.W:
            raise ValueError(
                f"batch width {self.batch_size_lon} exceeds longitude dimension {self.W}"
            )

        assert self.logger is not None, "Make sure the logger is set"
        self.logger.info(f"Spatial dimensions: {self.H} x {self.W}")
        self.logger.info(f"batch size: {self.batch_size_lat} x {self.batch_size_lon}")

        if self.constant_variables is not None and self.constants_file_path is not None:
            self.logger.info(f"Opening constant variables file: {constants_file_path}")
            # Open file
            ds_const = xr.open_dataset(self.constants_file_path).load()

            # Get the first time step (since there's only one) and drop time dimension
            ds_const = ds_const.isel(time=0)

            # Initialize constant variables tensor
            self.const_vars = np.zeros(
                (len(self.constant_variables), self.H, self.W)  # [channels, lat, lon]
            )

            for i, const_varname in enumerate(self.constant_variables):
                const_var = ds_const[const_varname]

                # Normalize if needed
                if const_varname != "lsm":
                    self.logger.info(f"Normalizing {const_varname}")
                    # Use xarray's weighted method (similar to your original code)
                    weighted_var = const_var.weighted(
                        np.cos(np.radians(ds_const.latitude))
                    )
                    mean_var = weighted_var.mean().values
                    std_var = weighted_var.std().values

                    self.logger.info(
                        f"{const_varname} - Weighted Mean: {mean_var:.4f}, Weighted Std: {std_var:.4f}"
                    )

                    # Normalize
                    self.const_vars[i] = ((const_var - mean_var) / std_var).values
                else:
                    self.const_vars[i] = const_var.values

            self.logger.info(f"Loaded constant variables: {self.constant_variables}")
            self.logger.info(f"Constant variables shape: {self.const_vars.shape}")

            # Close the dataset
            ds_const.close()
        else:
            self.const_vars = None
            self.logger.info("No constant variables provided")

        """
        # Load dynamic covariates if specified
        self.dynamic_covariate_data = None
        if self.dynamic_covariates:
            self.load_dynamic_covariates()
        """
        # Cache for loaded data
        self.loaded_dfs = loaded_dfs
        self.etime = len(self.loaded_dfs["time"])
        # ----------------------------------------------------------
        # 1. Extract time components from xarray
        # ----------------------------------------------------------
        self.time = self.loaded_dfs.time
        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.hour = self.time.dt.hour

        # ----------------------------------------------------------
        # 2. Normalized year
        # ----------------------------------------------------------
        year_np = ((self.year.to_numpy() - 1940) / 100).astype(self.np_dtype)
        self.year_norm = torch.from_numpy(year_np).to(self.torch_dtype)

        # ----------------------------------------------------------
        # 3. DOY & hour normalization
        # ----------------------------------------------------------
        if self.time_normalization == "linear":
            # Approximate DOY: (month-1)*30 + (day-1)
            doy_np = (
                ((self.month - 1.0) * 30 + (self.day - 1.0))
                .to_numpy()
                .astype(self.np_dtype)
            )
            hour_np = (self.hour.to_numpy() / 24.0).astype(self.np_dtype)

            self.doy_norm = torch.from_numpy((doy_np / 360).astype(self.np_dtype)).to(
                self.torch_dtype
            )
            self.hour_norm = torch.from_numpy(hour_np).to(self.torch_dtype)

        elif self.time_normalization == "cos_sin":
            date = pd.to_datetime(dict(year=self.year, month=self.month, day=self.day))
            doy = (date - datetime(2000, 1, 1)).dt.days

            self.doy_np = doy.to_numpy()
            self.unique_doys = np.unique(self.doy_np)
            self.doy_to_indices = {
                d: np.where(self.doy_np == d)[0] for d in self.unique_doys
            }

            doy_np = doy.to_numpy().astype(self.np_dtype)
            hour_np = self.hour.to_numpy().astype(self.np_dtype)

            self.doy_sin = torch.sin(
                2 * np.pi * torch.from_numpy(doy_np).to(self.torch_dtype) / 365.25
            )
            self.doy_cos = torch.cos(
                2 * np.pi * torch.from_numpy(doy_np).to(self.torch_dtype) / 365.25
            )
            self.hour_sin = torch.sin(
                2 * np.pi * torch.from_numpy(hour_np).to(self.torch_dtype) / 24.0
            )
            self.hour_cos = torch.cos(
                2 * np.pi * torch.from_numpy(hour_np).to(self.torch_dtype) / 24.0
            )

        else:
            raise ValueError("time_normalization must be 'linear' or 'cos_sin'")

        if self.mode == "validation":
            self.eval_slices = self.generate_evaluation_slices()
            # To Do: a key to add if all sbatch to taken or not
            self.sbatch = len(
                self.eval_slices
            )  # Set sbatch to exactly match number of slices
            self.logger.info(f"Evaluation mode: Generated {self.sbatch} spatial slices")
            # To Do: set time batches for validation (if inference is active or not)
            if self.run_type == "inference":
                self.time_batchs = np.arange(self.stime, self.etime, dtype=int)
                # self.time_batchs = np.linspace(
                #    self.etime // 3, self.etime * 2 // 3, self.tbatch, dtype=int
                # ) # 2 for debug, self.tbatch

            elif self.run_type == "inference_regional":
                # Regional inference mode:
                # Instead of tiling the full spatial domain (as in global validation),
                # we extract a single spatial window centered on a user-defined
                # geographical location (latitude, longitude).

                # Ensure that a target region center is provided
                assert (
                    self.region_center is not None
                ), "region_center must be provided for inference_regional mode"

                # Convert the requested (lat, lon) in degrees
                # to the nearest grid point indices,
                # because the dataset works with indices, not exact coordinates.
                lat_val, lon_val = self.region_center
                lat_idx, lon_idx = self.get_center_indices_from_latlon(lat_val, lon_val)

                # This center will be passed to extract_batch, which handles cyclic longitude
                self.inference_center = (lat_idx, lon_idx)

                # Only one spatial batch is needed (single regional window)
                # To Do: set sbatch if we use also n blocks for regional case
                self.sbatch = 1
                self.time_batchs = np.arange(self.stime, self.etime, dtype=int)
                # self.time_batchs = np.linspace(
                #    self.etime // 3, self.etime * 2 // 3, self.tbatch, dtype=int
                # ) # 2 for debug, self.tbatch

                self.logger.info(
                    f"Inference region mode activated at lat={lat_val}, lon={lon_val}"
                )

            else:
                self.time_batchs = np.linspace(
                    self.etime // 3, self.etime * 2 // 3, self.tbatch, dtype=int
                )
            self.logger.info(f"Validation time batches: {self.time_batchs}")
        else:
            # Training mode
            # To Do: set time batches for training (full or partial sampling)
            self.time_batchs = np.arange(self.stime, self.etime, dtype=int)
            self.logger.info(
                f"Training mode: sbatch={self.sbatch}, tbatch={self.tbatch}"
            )
            self.new_epoch()  # Initialize time batches for training

        self.center_tracker = []  # Will store spatial indices
        self.tindex_tracker = []  # Will store temporal indices

    def new_epoch(self):
        """
        Prepare for a new training epoch by generating new time batches.

        This method is called at the start of each training epoch to
        refresh the temporal and spatial sampling.
        """
        # The partial sampling by DOY is currently disabled
        # self.sample_time_steps_by_doy()
        # Also regenerate random centers for the new epoch
        self.random_centers = [None] * self.sbatch
        self.last_tbatch_index = -1

    def sample_time_steps_by_doy(self):
        """
        Sample time steps based on day-of-year (DOY) for multi-year continuity.

        This method selects unique DOYs from the available multi-year data
        and picks one random time index for each DOY.

        Raises
        ------
        ValueError
            If requested tbatch exceeds number of unique DOYs.
        """
        n = self.tbatch
        if n > len(self.unique_doys):
            raise ValueError(
                f"Requested tbatch={n}, but only {len(self.unique_doys)} unique days available."
            )

        # Select n unique DOYs from MULTI-YEAR continuous DOY sequence
        selected_doys = np.random.choice(self.unique_doys, size=n, replace=False)

        # For each DOY pick one random index where that DOY occurs
        self.time_batchs = np.array(
            [np.random.choice(self.doy_to_indices[d]) for d in selected_doys], dtype=int
        )

        if self.debug:
            self.logger.info(
                f"[DOY-batch] Selected DOYs: {selected_doys.tolist()} → indices: {self.time_batchs.tolist()}"
            )

    def sample_random_time_indices(self):
        """
        Generate random time indices for training.

        This method samples random time indices uniformly across the
        available time range.
        """
        # Only called from new_epoch which already checks mode == "train"
        self.time_batchs = np.random.randint(
            self.stime, self.etime - 1, size=self.tbatch
        )
        if self.debug:
            self.logger.info(f"Generated new training time batches: {self.time_batchs}")

    def load_dynamic_covariates(self):
        """Load dynamic covariates data."""
        self.dynamic_covariate_data = {}

        for covariate in self.dynamic_covariates:
            covariate_files = []
            for year in self.years:
                file_pattern = (
                    f"{self.dynamic_covariates_dir}/samples_{year}_{covariate}.nc"
                )
                matching_files = glob.glob(file_pattern)
                if matching_files:
                    covariate_files.extend(matching_files)

            if covariate_files:
                # Load and concatenate covariate data
                datasets = [xr.open_dataset(f) for f in covariate_files]
                combined_ds = xr.concat(datasets, dim="time")
                self.dynamic_covariate_data[covariate] = combined_ds

                # Close individual datasets
                for ds in datasets:
                    ds.close()

    def get_center_indices_from_latlon(self, lat_value, lon_value):
        """
        Convert geographic coordinates (latitude, longitude) to nearest grid indices.

        Parameters
        ----------
        lat_value : float
            Latitude in degrees.
        lon_value : float
            Longitude in degrees.

        Returns
        -------
        lat_idx : int
            Index of the closest latitude grid point.
        lon_idx : int
            Index of the closest longitude grid point.

        Notes
        -----
        - The dataset is defined on a discrete latitude–longitude grid.
        - Since spatial extraction operates on grid indices, the requested
          physical coordinates are mapped to the nearest available grid point.
        - This ensures consistency between user-defined locations and
          internal batch extraction logic.
        """

        # Retrieve latitude and longitude arrays from the dataset
        lat_array = self.loaded_dfs.latitude.values
        lon_array = self.loaded_dfs.longitude.values

        # Find the index of the grid point closest to the requested lat/lon
        lat_idx = np.abs(lat_array - lat_value).argmin()
        lon_idx = np.abs(lon_array - lon_value).argmin()

        return lat_idx, lon_idx

    def generate_random_batch_centers(self, n_batches):
        """
        Generate random (latitude, longitude) centers for batch sampling.

        Parameters
        ----------
        n_batches : int
            Number of random centers to generate.

        Returns
        -------
        centers : list of tuple
            List of (lat_center, lon_center) tuples.

        Notes
        -----
        - Latitude centers avoid poles to ensure full batch extraction.
        - Longitude centers can be any value due to cyclic wrapping.
        """
        centers = []
        half_lat = self.batch_size_lat // 2

        try:
            for _ in range(n_batches):
                # Latitude: avoid poles (non-cyclic)
                lat_center = np.random.randint(half_lat, self.H - half_lat)
                # Longitude: any (cyclic)
                lon_center = np.random.randint(0, self.W)
                centers.append((lat_center, lon_center))

            if self.debug:
                self.logger.info(
                    f"  [RandomBlockSampler]Generated {n_batches} random centers: {centers}"
                )
            return centers

        except Exception as e:
            self.logger.exception(
                f"[RandomBlockSampler] Error while generating random centers: {e}"
            )
            raise

    def generate_evaluation_slices(self):
        """
        Generate deterministic spatial slices for evaluation mode.

        Returns
        -------
        slices : list of tuple
            List of (lat_start, lat_end, lon_start, lon_end) tuples defining
            non-overlapping spatial blocks covering the entire domain.
        """
        n_blocks_lat = self.H // self.batch_size_lat
        n_blocks_lon = self.W // self.batch_size_lon

        # Create grid of block indices
        lat_idx, lon_idx = np.mgrid[0:n_blocks_lat, 0:n_blocks_lon]

        # Calculate slice boundaries
        lat_starts = (lat_idx * self.batch_size_lat).ravel()
        lon_starts = (lon_idx * self.batch_size_lon).ravel()

        lat_ends = lat_starts + self.batch_size_lat
        lon_ends = lon_starts + self.batch_size_lon

        # Create slices list
        slices = list(zip(lat_starts, lat_ends, lon_starts, lon_ends))

        self.logger.info(
            f"Generated {len(slices)} evaluation blocks "
            f"({n_blocks_lat} x {n_blocks_lon} grid)"
        )
        return slices

    def extract_batch(self, data, ilat, ilon):
        """
        Extract spatial batch centered at (ilat, ilon) with cyclic longitude.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input data with shape (..., H, W) where last two dimensions
            are latitude and longitude.
        ilat : int
            Latitude center index.
        ilon : int
            Longitude center index.

        Returns
        -------
        block : torch.Tensor or np.ndarray
            Extracted batch with shape (..., batch_size_lat, batch_size_lon).
        indices : tuple
            Tuple of (lat_start, lat_end, lon_start, lon_end) indices.

        Raises
        ------
        AssertionError
            If input tensor dimensions don't match grid dimensions or
            if indices are invalid.

        Notes
        -----
        - Longitude is treated as cyclic (wraps around 0-360°).
        - Latitude is non-cyclic (no wrapping at poles).
        - The function rolls the data to center the longitude and then
          extracts the appropriate slice.
        """
        try:
            H, W = data.shape[-2:]
            assert (
                H == self.H and W == self.W
            ), f"Input tensor shape ({H}, {W}) does not match sampler grid ({self.H}, {self.W})"

            half_lat = self.batch_size_lat // 2
            half_lon = self.batch_size_lon // 2

            # --- Compute latitude indices (non-cyclic) ---
            lat_start = ilat - half_lat
            lat_end = ilat + half_lat  # + 1  # +1 to include center line

            # --- Sanity check (should always hold given center generation logic) ---
            assert (
                0 <= lat_start <= self.H - self.batch_size_lat
            ), f"Invalid lat_start={lat_start}"
            assert (
                self.batch_size_lat <= lat_end <= self.H
            ), f"Invalid lat_end={lat_end}"

            # --- Longitude (cyclic) ---
            shift = W // 2 - ilon
            rolled = np.roll(data, shift=shift, axis=-1)

            lon_start = W // 2 - half_lon
            lon_end = W // 2 + half_lon  # + 1  # +1 to include center column

            block = rolled[..., lat_start:lat_end, lon_start:lon_end]

            if self.debug:
                # --- Logging ---
                self.logger.info(
                    f"  [extract_batch] ilat={ilat}, ilon={ilon}, "
                    f"  lat_range=({lat_start}:{lat_end}), lon_range=({lon_start}:{lon_end}), "
                    f"  shift={shift}, block_shape={tuple(block.shape)}"
                )

            # --- Warning for truncation near poles ---
            if block.shape[-2] != self.batch_size_lat:
                self.logger.warning(
                    f"  [extract_batch] Truncated block at ilat={ilat} "
                    f"  (lat size {block.shape[-2]} < {self.batch_size_lat})"
                )

            return block, (lat_start, lat_end, lon_start, lon_end)

        except Exception as e:
            self.logger.exception(
                f"[extract_batch] Unexpected error while extracting block: {e}"
            )
            raise

    def filter_batch(self, fine_patch, fine_block):
        """
        Apply Gaussian low-pass filtering for multi-scale processing.

        Parameters
        ----------
        fine_patch : np.ndarray
            Fine-resolution data of shape (C, H, W).
        fine_block : np.ndarray
            Reference block used to determine scaling factors.

        Returns
        -------
        fine_filtered : np.ndarray
            Filtered data of shape (C, H, W).

        Notes
        -----
        - Filters high-frequency components beyond the coarse grid's Nyquist.
        - Uses Gaussian filtering in the frequency domain.
        - Processes each channel independently.
        """

        # Output container
        fine_filtered = np.zeros_like(fine_patch)

        # ----------------------------------------------------------
        # Determine scaling between fine and coarse grids
        # ----------------------------------------------------------
        scale_factor_H = fine_block.shape[-2] / self.in_shape[0]
        scale_factor_W = fine_block.shape[-1] / self.in_shape[1]

        # ----------------------------------------------------------
        # Coarse-grid physical spacing and Nyquist limits
        # ----------------------------------------------------------
        dW_coarse = self.dW * scale_factor_W
        dH_coarse = self.dH * scale_factor_H

        f_nW_coarse = 1.0 / (2.0 * dW_coarse)
        f_nH_coarse = 1.0 / (2.0 * dH_coarse)

        # ----------------------------------------------------------
        # Apply filtering per channel
        # ----------------------------------------------------------
        C = fine_patch.shape[0]

        for c in range(C):
            try:
                img = fine_patch[c]
                filt = gaussian_filter(
                    img,
                    self.dW,
                    self.dH,
                    f_nW_coarse,
                    f_nH_coarse,
                    self.epsilon,
                    self.margin,
                )

                # Save the filtered channel
                fine_filtered[c] = filt

            except Exception as e:
                self.logger.exception(f"Filtering failed at channel {c}", e)
                raise

        return fine_filtered

    def normalize(self, data, stats, norm_type, var_name=None, data_type=None):
        """
        Normalize data using specified statistics and method.

        Parameters
        ----------
        data : torch.Tensor
            Input data to normalize.
        stats : object
            Statistics object with attributes: vmin, vmax, vmean, vstd,
            median, iqr, q1, q3.
        norm_type : str
            Normalization type: "minmax", "minmax_11", "standard", "robust",
            "log1p_minmax", "log1p_standard".
        var_name : str, optional
            Variable name for logging.
        data_type : str, optional
            Data type description for logging.

        Returns
        -------
        torch.Tensor
            Normalized data.

        Raises
        ------
        ValueError
            If norm_type is not supported.
        """
        # Create context string for logging
        context = ""
        if var_name is not None:
            context = f" for {var_name}"
        if data_type is not None:
            context += f" ({data_type})"

        if self.debug:
            self.logger.info(
                f"Normalizing{context} with type '{norm_type}'\n"
                f" └── Normalization stats:\n"
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
            vmin = torch.tensor(stats.vmin, dtype=data.dtype, device=data.device)
            vmax = torch.tensor(stats.vmax, dtype=data.dtype, device=data.device)
            denom = vmax - vmin
            if denom == 0:
                self.logger.warning("vmax == vmin, returning zeros.")
                return torch.zeros_like(data)
            return (data - vmin) / denom

        # ------------------ MIN-MAX [-1, 1] -----------------
        elif norm_type == "minmax_11":
            vmin = torch.tensor(stats.vmin, dtype=data.dtype, device=data.device)
            vmax = torch.tensor(stats.vmax, dtype=data.dtype, device=data.device)
            denom = vmax - vmin
            if denom == 0:
                self.logger.warning("vmax == vmin, returning zeros.")
                return torch.zeros_like(data)
            return 2 * (data - vmin) / denom - 1

        # ------------------ STANDARD -----------------
        elif norm_type == "standard":
            mean = torch.tensor(stats.vmean, dtype=data.dtype, device=data.device)
            std = torch.tensor(stats.vstd, dtype=data.dtype, device=data.device)
            if std == 0:
                self.logger.warning("vstd == 0, returning zeros.")
                return torch.zeros_like(data)
            return (data - mean) / std

        # ------------------ ROBUST -------------------
        elif norm_type == "robust":
            median = torch.tensor(stats.median, dtype=data.dtype, device=data.device)
            iqr = torch.tensor(stats.iqr, dtype=data.dtype, device=data.device)
            if iqr == 0:
                self.logger.warning("iqr == 0, returning zeros.")
                return torch.zeros_like(data)
            return (data - median) / iqr

        # ------------------ LOG1P + MIN-MAX ------------------
        elif norm_type == "log1p_minmax":
            data = torch.log1p(data)

            log_min = torch.tensor(stats.vmin, dtype=data.dtype, device=data.device)
            log_max = torch.tensor(stats.vmax, dtype=data.dtype, device=data.device)
            denom = log_max - log_min

            if denom == 0:
                self.logger.warning("log_max == log_min, returning zeros.")
                return torch.zeros_like(data)

            return (data - log_min) / denom

        # ------------------ LOG1P + STANDARD ------------------
        elif norm_type == "log1p_standard":
            data = torch.log1p(data)

            mean = torch.tensor(stats.vmean, dtype=data.dtype, device=data.device)
            std = torch.tensor(stats.vstd, dtype=data.dtype, device=data.device)

            if std == 0:
                self.logger.warning("log_std == 0, returning zeros.")
                return torch.zeros_like(data)

            return (data - mean) / std

        else:
            self.logger.error(f"Unsupported norm_type '{norm_type}'")
            raise ValueError(f"Unsupported norm_type '{norm_type}'")

    def normalize_time(self, tindex):
        """
        Return normalized time features for given time index.

        Parameters
        ----------
        tindex : int
            Time index.

        Returns
        -------
        dict
            Dictionary of normalized time features.

        Notes
        -----
        Features depend on time_normalization setting:
        - "linear": year_norm, doy_norm, hour_norm
        - "cos_sin": year_norm, doy_sin, doy_cos, hour_sin, hour_cos
        """

        out = {"year_norm": self.year_norm[tindex]}

        if self.time_normalization == "linear":
            out["doy_norm"] = self.doy_norm[tindex]
            out["hour_norm"] = self.hour_norm[tindex]

        elif self.time_normalization == "cos_sin":
            out["doy_sin"] = self.doy_sin[tindex]
            out["doy_cos"] = self.doy_cos[tindex]
            out["hour_sin"] = self.hour_sin[tindex]
            out["hour_cos"] = self.hour_cos[tindex]

        else:
            raise NotImplementedError(
                f"Time normalization '{self.time_normalization}' not implemented!"
            )

        return out

    def __len__(self):
        """
        Return total number of samples in the dataset.

        Returns
        -------
        int
            Total samples = number of time batches × number of spatial batches.
        """
        return len(self.time_batchs) * self.sbatch

    def __getitem__(self, index):
        """
        Get a single data sample.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        sample : dict
            Dictionary containing:

            - inputs: model input features including normalized coarse data,
              normalized coordinates, and constant variables.
            - targets: normalized residuals.
            - fine: original fine-resolution data.
            - coarse: coarse approximation before normalization.
            - coordinates: latitude and longitude coordinates for the batch.
            - time features: normalized temporal features.

        Raises
        ------
        IndexError
            If index is out of bounds.
        AssertionError
            If shape mismatches or other consistency checks fail.

        Notes
        -----
        - In training mode: random spatial and temporal sampling.
        - In validation mode: deterministic spatial slicing and fixed
          or sequential temporal sampling.
        - Applies multi-scale processing if apply_filter is True.
        - Normalizes data according to provided statistics.
        """
        if self.debug:
            self.logger.info("------------------- GET ITEM INFO -------------------")

        # Calculate spatial and temporal indices
        sindex = index % self.sbatch
        tbatch_index = index // self.sbatch

        if tbatch_index >= len(self.time_batchs):
            raise IndexError(f"Time batch index {tbatch_index} out of range")

        tindex = self.time_batchs[tbatch_index]
        self.tindex_tracker.append(tindex)

        if self.debug:
            self.logger.info(
                f"\nTorch batch index: {index}\n"
                f"Time block index: {tbatch_index}\n"
                f"Final time index (tindex): {tindex}\n"
                f"Spatial batch index (sindex): {sindex}\n"
            )

        # Load data
        full_data_org = self.loaded_dfs.isel(time=tindex)
        lat = full_data_org.latitude.values
        lon = full_data_org.longitude.values

        # Normalize to range [-1, 1] for better neural network input stability
        lat_norm = 2 * ((lat - lat.min()) / (lat.max() - lat.min())) - 1
        lon_norm = 2 * ((lon - lon.min()) / (lon.max() - lon.min())) - 1

        # 2D meshgrids of normalized latitude and longitude (shape: H x W)
        lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing="ij")

        sample = self.normalize_time(tindex)

        # Determine spatial sampling based on mode
        if self.mode == "validation":
            if self.run_type == "inference_regional":
                # Use a fixed geographic center instead of evaluation slices.
                # The region is centered on the user-defined inference location.

                # Retrieve the precomputed grid indices of the requested region center
                lat_center, lon_center = self.inference_center
                self.center_tracker.append((lat_center, lon_center))

                # Extract coordinate batches centered on the specified region
                lat_batch, lat_indices = self.extract_batch(
                    lat_grid, lat_center, lon_center
                )
                lat_start, lat_end, lon_start, lon_end = lat_indices

                lon_batch, lon_indices = self.extract_batch(
                    lon_grid, lat_center, lon_center
                )

                # Extract data for all variables into a NumPy array
                npfeatures_full = np.zeros([len(self.varnames_list), self.H, self.W])
                for var_name in self.varnames_list:
                    iv = self.index_mapping[var_name]
                    npfeatures_full[iv, :, :] = full_data_org[var_name].values

                # Extract spatial block centered on requested region
                fine_block, fine_indices = self.extract_batch(
                    npfeatures_full, lat_center, lon_center
                )

                if self.apply_filter:
                    fine_filtered_full = self.filter_batch(npfeatures_full, fine_block)
                    fine_filtered_block, _ = self.extract_batch(
                        fine_filtered_full, lat_center, lon_center
                    )

            else:  # validation, global inference
                # Ensure evaluation slices are available
                assert (
                    hasattr(self, "eval_slices") and self.eval_slices is not None
                ), "eval_slices not initialized for validation mode"
                assert (
                    len(self.eval_slices) > 0
                ), "eval_slices is empty for validation mode"

                # Deterministic spatial sampling for evaluation
                lat_start, lat_end, lon_start, lon_end = self.eval_slices[sindex]
                lat_center = lat_start + self.batch_size_lat // 2
                lon_center = lon_start + self.batch_size_lon // 2

                self.center_tracker.append((lat_center, lon_center))

                # Extract normalized coordinates for the batch
                lat_batch = lat_grid[lat_start:lat_end, lon_start:lon_end]
                lon_batch = lon_grid[lat_start:lat_end, lon_start:lon_end]

                # Extract data for all variables into the WHOLE spatial domain first (same as training)
                npfeatures_full = np.zeros([len(self.varnames_list), self.H, self.W])
                for i, var_name in enumerate(self.varnames_list):
                    npfeatures_full[i, :, :] = full_data_org[var_name].values

                # Extract the target batch for scaling determination
                fine_block = npfeatures_full[:, lat_start:lat_end, lon_start:lon_end]

                if self.apply_filter:
                    # Apply filter to the WHOLE domain (same as training logic)
                    fine_filtered_full = self.filter_batch(npfeatures_full, fine_block)

                    assert fine_filtered_full.shape == npfeatures_full.shape, (
                        f"Mismatch in shapes: fine_filtered has shape {fine_filtered_full.shape} "
                        f"but npfeatures has shape {npfeatures_full.shape}."
                    )

                    # Now extract the batch from filtered data
                    fine_filtered_block = fine_filtered_full[
                        :, lat_start:lat_end, lon_start:lon_end
                    ]
                    assert fine_filtered_block.shape == fine_block.shape, (
                        f"Mismatch in shapes: fine_filtered_block has shape {fine_filtered_block.shape} "
                        f"but fine_block has shape {fine_block.shape}."
                    )

                if self.debug:
                    self.logger.info(
                        f"  Evaluation mode: slice lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]\n"
                        f"  Full domain shape: {npfeatures_full.shape}\n"
                        f"  Batch shape: {fine_block.shape}"
                    )

        else:
            # Random spatial sampling for training
            if self.last_tbatch_index != tbatch_index:
                self.random_centers = self.generate_random_batch_centers(self.sbatch)
                self.last_tbatch_index = tbatch_index

            assert (
                self.random_centers[sindex] is not None
            ), f"Random center at index {sindex} has not been generated yet."
            lat_center, lon_center = self.random_centers[sindex]
            self.center_tracker.append((lat_center, lon_center))

            if self.debug:
                self.logger.info(f"  Training mode: center ({lat_center},{lon_center})")

            lat_batch, lat_indices = self.extract_batch(
                lat_grid, lat_center, lon_center
            )
            lat_start, lat_end, lon_start, lon_end = lat_indices
            lon_batch, lon_indices = self.extract_batch(
                lon_grid, lat_center, lon_center
            )
            assert lat_indices == lon_indices, (
                f"Indices mismatch for same center (lat_center={lat_center}, lon_center={lon_center}):\n"
                f"  lat_indices: {lat_indices}\n"
                f"  lon_indices: {lon_indices}"
            )

            # Extract data for all variables into a NumPy array (full domain)
            npfeatures_full = np.zeros([len(self.varnames_list), self.H, self.W])
            for var_name in self.varnames_list:
                iv = self.index_mapping[var_name]
                npfeatures_full[iv, :, :] = full_data_org[var_name].values

            # Extract the batch and filter (full domain filtering like validation)
            fine_block, fine_indices = self.extract_batch(
                npfeatures_full, lat_center, lon_center
            )
            assert fine_indices == lat_indices, (
                f"Indices mismatch between data and coordinate extractions:\n"
                f"  lat/lon indices: {lat_indices}\n"
                f"  data indices: {fine_indices}"
            )

            if self.apply_filter:
                fine_filtered_full = self.filter_batch(npfeatures_full, fine_block)

                assert fine_filtered_full.shape == npfeatures_full.shape, (
                    f"Mismatch in shapes: fine_filtered has shape {fine_filtered_full.shape} "
                    f"but npfeatures has shape {npfeatures_full.shape}."
                )

                fine_filtered_block, filtered_indices = self.extract_batch(
                    fine_filtered_full, lat_center, lon_center
                )
                assert filtered_indices == lat_indices, (
                    f"Indices mismatch after filtering:\n"
                    f"  original indices: {lat_indices}\n"
                    f"  filtered indices: {filtered_indices}"
                )

                assert fine_filtered_block.shape == fine_block.shape, (
                    f"Mismatch in shapes: fine_filtered_block has shape {fine_filtered_block.shape} "
                    f"but fine_block has shape {fine_block.shape}."
                )

        # Common processing for both modes
        if self.apply_filter:
            coarse = coarse_down_up(fine_filtered_block, fine_block)
        else:
            coarse = coarse_down_up(fine_block, fine_block)

        # Convert fine data to torch tensor and initialize normalized container
        fine_block = torch.from_numpy(fine_block).to(self.torch_dtype)
        fine_block_norm = torch.zeros_like(fine_block)

        #  Ensure coarse are torch tensors with correct dtype
        if isinstance(coarse, np.ndarray):
            coarse = torch.from_numpy(coarse)
        elif not torch.is_tensor(coarse):
            raise TypeError(f"Unexpected type for coarse: {type(coarse)}")

        coarse = coarse.to(self.torch_dtype)

        coarse_norm = torch.zeros_like(coarse)

        # If filtering is enabled, also prepare the filtered fine field
        if self.apply_filter:
            fine_filtered_block = torch.from_numpy(fine_filtered_block).to(
                self.torch_dtype
            )

        # Normalize fine and coarse fields using variable-specific statistics.
        # Statistics are read from the JSON file and are independent for fine and coarse data.
        # For log-based normalizations, the stored statistics correspond to log1p(fine)/log1p(coarse).
        for var_name in self.varnames_list:
            iv = self.index_mapping[var_name]
            norm_type = self.normalization_type[var_name]
            # Select appropriate statistics depending on the normalization type
            if norm_type.startswith("log1p"):
                stats_fine = self.norm_mapping[f"{var_name}_fine_log"]
                stats_coarse = self.norm_mapping[f"{var_name}_coarse_log"]
            else:
                stats_fine = self.norm_mapping[f"{var_name}_fine"]
                stats_coarse = self.norm_mapping[f"{var_name}_coarse"]

            # Normalize coarse field
            coarse_norm[iv] = self.normalize(
                coarse[iv],
                stats_coarse,
                norm_type,
                var_name=var_name,
                data_type="coarse",
            )

            # Normalize filtered fine field if applicable
            if self.apply_filter:
                fine_block_norm[iv] = self.normalize(
                    fine_filtered_block[iv],
                    stats_fine,
                    norm_type,
                    var_name=var_name,
                    data_type="fine",
                )
            # Normalize fine field
            else:
                fine_block_norm[iv] = self.normalize(
                    fine_block[iv],
                    stats_fine,
                    norm_type,
                    var_name=var_name,
                    data_type="fine",
                )

        # residual is defined in normalized space
        residual = fine_block_norm - coarse_norm

        expected_shape = (
            len(self.varnames_list),
            self.batch_size_lat,
            self.batch_size_lon,
        )
        assert (
            coarse.shape == expected_shape
        ), f"coarse shape {coarse.shape} != expected {expected_shape}"
        assert (
            residual.shape == expected_shape
        ), f"residual shape {residual.shape} != expected {expected_shape}"

        # Ensure spatial batches are torch tensors with correct dtype
        lat_batch = (
            torch.from_numpy(lat_batch)
            if isinstance(lat_batch, np.ndarray)
            else lat_batch
        )
        lon_batch = (
            torch.from_numpy(lon_batch)
            if isinstance(lon_batch, np.ndarray)
            else lon_batch
        )
        lat_batch = lat_batch.to(self.torch_dtype)
        lon_batch = lon_batch.to(self.torch_dtype)

        # Ensure residual are torch tensors with correct dtype
        residual = (
            torch.from_numpy(residual) if isinstance(residual, np.ndarray) else residual
        )
        residual = residual.to(self.torch_dtype)

        # Ensure coarse_norm/fine_norm are torch tensors with correct dtype
        coarse_norm = (
            torch.from_numpy(coarse_norm)
            if isinstance(coarse_norm, np.ndarray)
            else coarse_norm
        )
        fine_block_norm = (
            torch.from_numpy(fine_block_norm)
            if isinstance(fine_block_norm, np.ndarray)
            else fine_block_norm
        )
        coarse_norm = coarse_norm.to(self.torch_dtype)
        fine_block_norm = fine_block_norm.to(self.torch_dtype)

        feature = torch.cat(
            [coarse_norm, lat_batch.unsqueeze(0), lon_batch.unsqueeze(0)], dim=0
        )

        if self.debug:
            self.logger.info(f"  Feature composition before constants: {feature.shape}")

        if self.constant_variables is not None:
            assert self.const_vars is not None, (
                f"Constant variables {self.constant_variables} were specified "
                f"but const_vars could not be loaded. Please check the file path and variable names."
            )

            if self.mode == "validation":
                if self.run_type == "inference_regional":
                    # For inference_regional, use extract_batch
                    const_batch, const_indices = self.extract_batch(
                        self.const_vars, lat_center, lon_center
                    )
                    assert const_indices == lat_indices, (
                        f"Indices mismatch for constant variables:\n"
                        f"  coordinate indices: {lat_indices}\n"
                        f"  constant var indices: {const_indices}"
                    )

                else:  # validation, inference_global
                    # For evaluation, use direct slicing
                    const_batch = self.const_vars[
                        :, lat_start:lat_end, lon_start:lon_end
                    ]
            else:
                # For training, use extract_batch
                const_batch, const_indices = self.extract_batch(
                    self.const_vars, lat_center, lon_center
                )
                assert const_indices == lat_indices, (
                    f"Indices mismatch for constant variables:\n"
                    f"  coordinate indices: {lat_indices}\n"
                    f"  constant var indices: {const_indices}"
                )

            const_batch = (
                torch.from_numpy(const_batch)
                if isinstance(const_batch, np.ndarray)
                else const_batch
            )
            const_batch = const_batch.to(self.torch_dtype)

            if self.debug:
                self.logger.info(f"  Constant batch shape: {const_batch.shape}")

            feature = torch.cat([feature, const_batch], dim=0)

            if self.debug:
                self.logger.info(
                    f"  Feature shape after adding constants: {feature.shape}"
                )

        if self.run_type == "inference_regional":
            # In regional inference mode, we extract a spatial window centered
            # on a user-defined (lat_center, lon_center)
            # Therefore, longitude requires special handling to avoid
            # discontinuities when crossing the dateline

            lat_vals = lat[lat_start:lat_end]

            # Recompute the same shift used inside extract_batch
            shift = self.W // 2 - lon_center
            lon_rolled = np.roll(lon, shift=shift)
            lon_vals = lon_rolled[lon_start:lon_end]
            lon_vals = ((lon_vals + 180) % 360) - 180

        else:
            lat_vals = lat[lat_start:lat_end]
            lon_vals = lon[lon_start:lon_end]

        sample.update(
            {
                "inputs": feature,  # model inputs (coarse_norm + coordinates + constants)
                "targets": residual,  # residual in normalized space (model target)
                "fine": fine_block,  # fine-resolution physical data (for diagnostics)
                "coarse": coarse,  # coarse physical data (baselinefor diagnostics)
                # "coarse_norm": coarse_norm, # coarse normalized data (for reconstruction at validation)
                "corrdinates": {
                    "lat": torch.from_numpy(lat_vals).to(self.torch_dtype),
                    "lon": torch.from_numpy(lon_vals).to(self.torch_dtype),
                },
            }
        )

        if self.debug:
            self.logger.info("------------------------------------------------------")

        return sample


# ============================================================================
# Test Utilities
# ============================================================================


def create_dummy_netcdf(temp_dir, year=2020, has_constants=False):
    """Create a dummy NetCDF file for testing."""
    lat = np.linspace(-90, 90, 36)
    lon = np.linspace(-180, 180, 72)
    time = pd.date_range(f"{year}-01-01", f"{year}-01-03", freq="6h")

    data_vars = {
        "VAR_2T": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 10 + 285,
        ),
        "VAR_10U": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 3 + 0.5,
        ),
        "VAR_10V": (
            ("time", "latitude", "longitude"),
            np.random.randn(len(time), len(lat), len(lon)) * 3 - 0.1,
        ),
        "VAR_TP": (
            ("time", "latitude", "longitude"),
            np.abs(np.random.randn(len(time), len(lat), len(lon)) * 0.0002 + 0.00009),
        ),
    }

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": time,
            "latitude": lat,
            "longitude": lon,
        },
    )

    if has_constants:
        const_ds = xr.Dataset(
            {
                "z": (
                    ("latitude", "longitude"),
                    np.random.randn(len(lat), len(lon)) * 100 + 500,
                ),
                "lsm": (
                    ("latitude", "longitude"),
                    np.random.randint(0, 2, (len(lat), len(lon))),
                ),
            },
            coords={"latitude": lat, "longitude": lon, "time": [time[0]]},
        )
        const_path = os.path.join(temp_dir, "constants.nc")
        const_ds.to_netcdf(const_path)
        return ds, const_path

    return ds


def create_dummy_statistics_json(temp_dir):
    """Create dummy statistics.json file for testing."""
    stats_dict = {
        "VAR_2T_coarse": {"vmean": 285.04, "vstd": 12.7438},
        "VAR_2T_residual": {"vmean": -0.000094627, "vstd": 1.6042},
        "VAR_2T_fine": {"vmean": 286.31, "vstd": 12.6325},
        "VAR_10U_coarse": {"vmean": 0.44536, "vstd": 3.4649},
        "VAR_10U_residual": {"vmean": -0.0013833, "vstd": 1.0221},
        "VAR_10U_fine": {"vmean": 0.36626, "vstd": 3.4527},
        "VAR_10V_coarse": {"vmean": -0.11892, "vstd": 3.7420},
        "VAR_10V_residual": {"vmean": -0.0015548, "vstd": 1.0384},
        "VAR_10V_fine": {"vmean": -0.05726, "vstd": 3.6996},
        "VAR_TP_coarse": {"vmean": 0.000094189, "vstd": 0.00026393},
        "VAR_TP_residual": {"vmean": -0.000000040417, "vstd": 0.00028678},
        "VAR_TP_fine": {"vmean": 0.000097360, "vstd": 0.00044533},
    }

    stats_path = os.path.join(temp_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f)

    return stats_path


# ============================================================================
# Unit Tests for DataPreprocessor
# ============================================================================


class TestDataPreprocessor(unittest.TestCase):
    """Unit tests for DataPreprocessor class."""

    def __init__(self, methodName="runTest", logger=None):
        super().__init__(methodName)
        self.logger = logger

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        if self.logger:
            self.logger.info(f"Test setup - created temp directory: {self.temp_dir}")

        # Create dummy dataset
        self.ds, self.const_path = create_dummy_netcdf(
            self.temp_dir, year=2020, has_constants=True
        )

        # Create statistics JSON
        create_dummy_statistics_json(self.temp_dir)

        # Setup configuration
        self.years = [2020]
        self.varnames_list = ["VAR_2T", "VAR_10U", "VAR_10V", "VAR_TP"]
        self.units_list = ["K", "m/s", "m/s", "m"]
        self.in_shape = (8, 16)
        self.batch_size_lat = 32
        self.batch_size_lon = 32

        # Compute steps
        self.norm_mapping, self.steps = stats(
            self.ds, self.logger, input_dir=self.temp_dir
        )

        # Create index mapping
        self.index_mapping = {var: i for i, var in enumerate(self.varnames_list)}

        # Normalization types
        self.normalization_type = {var: "standard" for var in self.varnames_list}

        # Constant variables
        self.constant_variables = ["z", "lsm"]

        if self.logger:
            self.logger.info(
                "Test setup complete - DataPreprocessor test fixtures ready"
            )

    # ------------------------------------------------------------------------
    # Stats Function Tests
    # ------------------------------------------------------------------------

    def test_stats_computation_without_existing_json(self):
        """Test stats computation when no statistics.json exists."""
        if self.logger:
            self.logger.info("Testing stats computation without existing JSON")

        # Create a fresh dataset and directory without stats.json
        temp_dir_no_stats = tempfile.mkdtemp()
        ds_no_stats = create_dummy_netcdf(temp_dir_no_stats, year=2020)

        norm_mapping = {}
        result_norm, steps = stats(
            ds_no_stats,
            self.logger,
            input_dir=temp_dir_no_stats,
            norm_mapping=norm_mapping,
        )

        # Check that norm_mapping was populated with constants
        self.assertIn("VAR_2T_coarse", result_norm)
        self.assertIn("VAR_2T_residual", result_norm)
        self.assertEqual(result_norm["VAR_2T_coarse"].vmean, 285.04)
        self.assertEqual(result_norm["VAR_2T_coarse"].vstd, 12.7438)

        # Check coordinate stats
        self.assertIn("latitude", result_norm)
        self.assertIn("longitude", result_norm)

        # Check steps
        self.assertEqual(steps["latitude"], 36)
        self.assertEqual(steps["longitude"], 72)
        self.assertEqual(steps["time"], 9)

        shutil.rmtree(temp_dir_no_stats)

        if self.logger:
            self.logger.info("✅ Stats computation without JSON test passed")

    def test_stats_computation_with_existing_json(self):
        """Test stats computation when statistics.json exists."""
        if self.logger:
            self.logger.info("Testing stats computation with existing JSON")

        norm_mapping = {}
        result_norm, _ = stats(
            self.ds, self.logger, input_dir=self.temp_dir, norm_mapping=norm_mapping
        )

        # Check that stats were loaded from JSON
        self.assertEqual(result_norm["VAR_2T_coarse"].vmean, 285.04)
        self.assertEqual(result_norm["VAR_2T_residual"].vstd, 1.6042)

        if self.logger:
            self.logger.info("✅ Stats computation with JSON test passed")

    # ------------------------------------------------------------------------
    # CoarseDownUp Function Tests
    # ------------------------------------------------------------------------
    def test_coarse_down_up_with_torch_tensors(self):
        """Test coarse_down_up with torch tensor inputs."""
        if self.logger:
            self.logger.info("Testing coarse_down_up with torch tensors")

        channels = 4
        h_fine, w_fine = 144, 144
        coarse_shape = (16, 32)

        fine_filtered = torch.randn(channels, h_fine, w_fine)
        fine_batch = torch.randn(channels, h_fine, w_fine)

        coarse_up = coarse_down_up(fine_filtered, fine_batch, input_shape=coarse_shape)

        self.assertEqual(coarse_up.shape, (channels, h_fine, w_fine))

        if self.logger:
            self.logger.info(
                f"✅ Coarse_down_up test passed - output shapes: {coarse_up.shape}"
            )

    def test_coarse_down_up_with_numpy_arrays(self):
        """Test coarse_down_up with numpy array inputs."""
        if self.logger:
            self.logger.info("Testing coarse_down_up with numpy arrays")

        channels = 4
        h_fine, w_fine = 144, 144

        fine_filtered = np.random.randn(channels, h_fine, w_fine).astype(np.float32)
        fine_batch = np.random.randn(channels, h_fine, w_fine).astype(np.float32)

        coarse_up = coarse_down_up(fine_filtered, fine_batch)

        self.assertIsInstance(coarse_up, torch.Tensor)
        self.assertEqual(coarse_up.shape, (channels, h_fine, w_fine))

        if self.logger:
            self.logger.info("✅ Coarse_down_up with numpy arrays test passed")

    # ------------------------------------------------------------------------
    # GaussianFilter Function Tests
    # ------------------------------------------------------------------------

    def test_gaussian_filter_basic(self):
        """Test basic Gaussian filtering operation."""
        if self.logger:
            self.logger.info("Testing Gaussian filter basic operation")

        H, W = 144, 144
        image = np.random.randn(H, W).astype(np.float32)
        dH = dW = 0.5
        cutoff_H = cutoff_W = 0.1

        filtered = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.01, margin=8
        )

        self.assertEqual(filtered.shape, image.shape)
        self.assertLess(np.var(filtered), np.var(image))

        if self.logger:
            self.logger.info(
                f"✅ Gaussian filter test passed - variance reduced from {np.var(image):.4f} to {np.var(filtered):.4f}"
            )

    def test_gaussian_filter_different_epsilon(self):
        """Test Gaussian filter with different epsilon values."""
        if self.logger:
            self.logger.info("Testing Gaussian filter with different epsilon values")

        H, W = 144, 144
        image = np.random.randn(H, W).astype(np.float32)
        dH = dW = 0.5
        cutoff_H = cutoff_W = 0.1

        filtered_smooth = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.001, margin=8
        )
        filtered_sharp = gaussian_filter(
            image, dW, dH, cutoff_W, cutoff_H, epsilon=0.1, margin=8
        )

        self.assertGreater(np.var(filtered_sharp), np.var(filtered_smooth))

        if self.logger:
            self.logger.info(
                f"✅ Gaussian filter epsilon test passed - sharp var: {np.var(filtered_sharp):.4f}, smooth var: {np.var(filtered_smooth):.4f}"
            )

    # ------------------------------------------------------------------------
    # DataPreprocessor Initialization Tests
    # ------------------------------------------------------------------------

    def test_preprocessor_initialization_train_mode(self):
        """Test DataPreprocessor initialization in train mode."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor initialization - TRAIN mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify dimensions
        self.assertEqual(preprocessor.H, 36)
        self.assertEqual(preprocessor.W, 72)
        self.assertEqual(len(preprocessor.time_batchs), 9)  # Full time range
        self.assertEqual(preprocessor.sbatch, 4)
        self.assertEqual(preprocessor.tbatch, 2)

        # Verify constant variables
        self.assertIsNotNone(preprocessor.const_vars)
        self.assertEqual(preprocessor.const_vars.shape, (2, 36, 72))

        # Verify time normalization
        self.assertIsNotNone(preprocessor.year_norm)
        self.assertIsNotNone(preprocessor.doy_norm)
        self.assertIsNotNone(preprocessor.hour_norm)

        if self.logger:
            self.logger.info(
                f"✅ Train mode initialization test passed - H={preprocessor.H}, W={preprocessor.W}"
            )

    def test_preprocessor_initialization_validation_mode(self):
        """Test DataPreprocessor initialization in validation mode."""
        if self.logger:
            self.logger.info(
                "Testing DataPreprocessor initialization - VALIDATION mode"
            )

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify evaluation slices
        self.assertIsNotNone(preprocessor.eval_slices)
        expected_slices = (36 // 32) * (72 // 32)  # 1 * 2 = 2
        self.assertEqual(len(preprocessor.eval_slices), expected_slices)
        self.assertEqual(preprocessor.sbatch, expected_slices)

        if self.logger:
            self.logger.info(
                f"✅ Validation mode initialization test passed - generated {len(preprocessor.eval_slices)} evaluation slices"
            )

    def test_preprocessor_initialization_cos_sin_normalization(self):
        """Test DataPreprocessor with cos_sin time normalization."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor with cos_sin time normalization")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="cos_sin",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Verify time normalization features
        self.assertIsNotNone(preprocessor.doy_sin)
        self.assertIsNotNone(preprocessor.doy_cos)
        self.assertIsNotNone(preprocessor.hour_sin)
        self.assertIsNotNone(preprocessor.hour_cos)

        if self.logger:
            self.logger.info("✅ Cos_sin time normalization test passed")

    def test_preprocessor_batch_size_validation(self):
        """Test that batch size validation raises appropriate errors."""
        if self.logger:
            self.logger.info("Testing batch size validation")

        with self.assertRaises(ValueError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=50,  # > 36
                batch_size_lon=self.batch_size_lon,
                steps=self.steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="linear",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Batch size validation test passed")

    # ------------------------------------------------------------------------
    # DataPreprocessor Method Tests
    # ------------------------------------------------------------------------

    def test_new_epoch_method(self):
        """Test new_epoch method resets random centers."""
        if self.logger:
            self.logger.info("Testing new_epoch method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Set some random centers
        preprocessor.random_centers = [(10, 20), (15, 30), (20, 40), (25, 50)]
        preprocessor.new_epoch()

        # Verify centers were reset
        self.assertEqual(preprocessor.random_centers, [None] * 4)
        self.assertEqual(preprocessor.last_tbatch_index, -1)

        if self.logger:
            self.logger.info("✅ New_epoch method test passed")

    def test_generate_random_batch_centers(self):
        """Test random batch centers generation."""
        if self.logger:
            self.logger.info("Testing generate_random_batch_centers")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        centers = preprocessor.generate_random_batch_centers(5)

        self.assertEqual(len(centers), 5)

        half_lat = preprocessor.batch_size_lat // 2
        for lat_center, lon_center in centers:
            self.assertGreaterEqual(lat_center, half_lat)
            self.assertLess(lat_center, preprocessor.H - half_lat)
            self.assertGreaterEqual(lon_center, 0)
            self.assertLess(lon_center, preprocessor.W)

        if self.logger:
            self.logger.info(
                f"✅ Generate random batch centers test passed - generated {len(centers)} centers"
            )

    def test_extract_batch(self):
        """Test spatial batch extraction."""
        if self.logger:
            self.logger.info("Testing extract_batch method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        # Create dummy data
        data = np.random.randn(36, 72)
        ilat, ilon = 18, 36

        block, indices = preprocessor.extract_batch(data, ilat, ilon)
        lat_start, lat_end, lon_start, lon_end = indices

        self.assertEqual(block.shape[-2:], (32, 32))
        self.assertEqual(lat_end - lat_start, 32)
        self.assertEqual(lon_end - lon_start, 32)

        if self.logger:
            self.logger.info(
                f"✅ Extract_batch test passed - block shape: {block.shape}"
            )

    def test_generate_evaluation_slices(self):
        """Test evaluation slices generation."""
        if self.logger:
            self.logger.info("Testing generate_evaluation_slices")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        slices = preprocessor.generate_evaluation_slices()

        n_blocks_lat = 36 // 32
        n_blocks_lon = 72 // 32
        self.assertEqual(len(slices), n_blocks_lat * n_blocks_lon)

        if self.logger:
            self.logger.info(
                f"✅ Generate evaluation slices test passed - created {len(slices)} slices"
            )

    # ------------------------------------------------------------------------
    # Normalization Tests
    # ------------------------------------------------------------------------

    def test_normalize_methods(self):
        """Test different normalization methods."""
        if self.logger:
            self.logger.info("Testing normalize methods")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        stats_obj = EasyDict(
            {
                "vmin": 0.0,
                "vmax": 100.0,
                "vmean": 50.0,
                "vstd": 25.0,
                "median": 50.0,
                "iqr": 30.0,
                "q1": 35.0,
                "q3": 65.0,
            }
        )

        data = torch.tensor([25.0, 50.0, 75.0])

        # Test minmax
        result_minmax = preprocessor.normalize(data, stats_obj, "minmax")
        expected_minmax = torch.tensor([0.25, 0.5, 0.75])
        torch.testing.assert_close(result_minmax, expected_minmax)

        # Test standard
        result_standard = preprocessor.normalize(data, stats_obj, "standard")
        expected_standard = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(result_standard, expected_standard)

        # Test invalid
        with self.assertRaises(ValueError):
            preprocessor.normalize(data, stats_obj, "invalid_type")

        if self.logger:
            self.logger.info("✅ Normalize methods test passed")

    def test_normalize_time_linear(self):
        """Test time normalization with linear method."""
        if self.logger:
            self.logger.info("Testing normalize_time with linear method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        time_features = preprocessor.normalize_time(0)

        self.assertIn("year_norm", time_features)
        self.assertIn("doy_norm", time_features)
        self.assertIn("hour_norm", time_features)

        if self.logger:
            self.logger.info("✅ Normalize_time linear test passed")

    def test_normalize_time_cos_sin(self):
        """Test time normalization with cos_sin method."""
        if self.logger:
            self.logger.info("Testing normalize_time with cos_sin method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="cos_sin",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        time_features = preprocessor.normalize_time(0)

        self.assertIn("year_norm", time_features)
        self.assertIn("doy_sin", time_features)
        self.assertIn("doy_cos", time_features)
        self.assertIn("hour_sin", time_features)
        self.assertIn("hour_cos", time_features)

        # Verify sin^2 + cos^2 ≈ 1
        sin_sq_cos_sq = time_features["doy_sin"] ** 2 + time_features["doy_cos"] ** 2
        torch.testing.assert_close(sin_sq_cos_sq, torch.ones_like(sin_sq_cos_sq))

        if self.logger:
            self.logger.info("✅ Normalize_time cos_sin test passed")

    # ------------------------------------------------------------------------
    # Dataset Methods Tests
    # ------------------------------------------------------------------------

    def test_dataset_len_method(self):
        """Test __len__ method."""
        if self.logger:
            self.logger.info("Testing __len__ method")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=3,
            sbatch=5,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        self.assertEqual(len(preprocessor), 9 * 5)  # all time steps * sbatch

        if self.logger:
            self.logger.info(
                f"✅ __len__ method test passed - length = {len(preprocessor)}"
            )

    def test_getitem_train_mode(self):
        """Test __getitem__ method in train mode."""
        if self.logger:
            self.logger.info("Testing __getitem__ in train mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=2,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify sample structure
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)
        self.assertIn("fine", sample)
        self.assertIn("coarse", sample)
        self.assertIn("corrdinates", sample)

        # Verify shapes
        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["fine"].shape, (n_vars, 32, 32))
        self.assertEqual(sample["coarse"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ __getitem__ train mode test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_getitem_validation_mode(self):
        """Test __getitem__ method in validation mode."""
        if self.logger:
            self.logger.info("Testing __getitem__ in validation mode")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="validation",
            run_type="validation",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify sample structure
        self.assertIn("inputs", sample)
        self.assertIn("targets", sample)

        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ __getitem__ validation mode test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_getitem_with_filter_enabled(self):
        """Test __getitem__ with apply_filter=True."""
        if self.logger:
            self.logger.info("Testing __getitem__ with filter enabled")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=self.const_path,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=2,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=self.constant_variables,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=True,
            logger=self.logger,
        )

        sample = preprocessor[0]

        # Verify shapes still correct with filtering
        n_vars = len(self.varnames_list)
        n_const = len(self.constant_variables)
        expected_input_channels = n_vars + 2 + n_const

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))
        self.assertEqual(sample["targets"].shape, (n_vars, 32, 32))

        if self.logger:
            self.logger.info("✅ __getitem__ with filter enabled test passed")

    def test_invalid_coordinate_handling(self):
        """Test handling of invalid coordinate specifications."""
        if self.logger:
            self.logger.info("Testing invalid coordinate handling")

        # Remove latitude from steps
        bad_steps = EasyDict(self.steps)
        delattr(bad_steps, "latitude")
        delattr(bad_steps, "d_latitude")

        with self.assertRaises(AssertionError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=self.batch_size_lat,
                batch_size_lon=self.batch_size_lon,
                steps=bad_steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="linear",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Invalid coordinate handling test passed")

    def test_preprocessor_without_constants(self):
        """Test DataPreprocessor without constant variables."""
        if self.logger:
            self.logger.info("Testing DataPreprocessor without constants")

        preprocessor = DataPreprocessor(
            years=self.years,
            loaded_dfs=self.ds,
            constants_file_path=None,
            varnames_list=self.varnames_list,
            units_list=self.units_list,
            in_shape=self.in_shape,
            batch_size_lat=self.batch_size_lat,
            batch_size_lon=self.batch_size_lon,
            steps=self.steps,
            tbatch=2,
            sbatch=4,
            debug=True,
            mode="train",
            run_type="train",
            time_normalization="linear",
            norm_mapping=self.norm_mapping,
            index_mapping=self.index_mapping,
            normalization_type=self.normalization_type,
            constant_variables=None,
            epsilon=0.02,
            margin=8,
            dtype=(torch.float32, np.float32),
            apply_filter=False,
            logger=self.logger,
        )

        self.assertIsNone(preprocessor.const_vars)

        sample = preprocessor[0]

        # Without constants, inputs should be n_vars + 2
        n_vars = len(self.varnames_list)
        expected_input_channels = n_vars + 2

        self.assertEqual(sample["inputs"].shape, (expected_input_channels, 32, 32))

        if self.logger:
            self.logger.info(
                f"✅ No constants test passed - inputs shape: {sample['inputs'].shape}"
            )

    def test_invalid_time_normalization(self):
        """Test invalid time normalization method."""
        if self.logger:
            self.logger.info("Testing invalid time normalization")

        with self.assertRaises(ValueError):
            DataPreprocessor(
                years=self.years,
                loaded_dfs=self.ds,
                constants_file_path=self.const_path,
                varnames_list=self.varnames_list,
                units_list=self.units_list,
                in_shape=self.in_shape,
                batch_size_lat=self.batch_size_lat,
                batch_size_lon=self.batch_size_lon,
                steps=self.steps,
                tbatch=2,
                sbatch=4,
                debug=True,
                mode="train",
                run_type="train",
                time_normalization="invalid_method",
                norm_mapping=self.norm_mapping,
                index_mapping=self.index_mapping,
                normalization_type=self.normalization_type,
                constant_variables=self.constant_variables,
                logger=self.logger,
            )

        if self.logger:
            self.logger.info("✅ Invalid time normalization test passed")

    def tearDown(self):
        """Clean up after tests."""

        shutil.rmtree(self.temp_dir)
        if self.logger:
            self.logger.info(f"Test teardown - removed temp directory: {self.temp_dir}")
