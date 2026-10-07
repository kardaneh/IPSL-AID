# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston, Pierre Chapel, Mehdi Lakbar, and others
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

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


def stats(ds, logger, input_dir, norm_mapping=dict()):
    """
    Load normalization statistics and compute coordinate metadata for a NetCDF dataset.

    This function loads normalization statistics from a JSON file if available.
    If no statistics file is found, it falls back to predefined constants for
    fine and coarse variables only.

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
    - If statistics.json is found, all statistics are loaded as-is.
    - If not found, fallback constants are used for fine and coarse variables only.
    """

    # stats_file_path = os.path.join(output_dir, "statistics.json")

    logger.info("Starting statistics computation...")
    steps = EasyDict()

    # ds = ds.drop_vars(["number", "expver"], errors="ignore")

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
            "VAR_2T_fine": {"vmean": 286.3158874511719, "vstd": 12.632543563842773},
            "VAR_10U_fine": {"vmean": 0.36626559495925903, "vstd": 3.4527664184570312},
            "VAR_10V_fine": {"vmean": -0.05726669356226921, "vstd": 3.699695110321045},
            "VAR_TP_fine": {
                "vmean": 9.736002539284527e-05,
                "vstd": 0.0004453345318324864,
            },
            "VAR_D2M_fine": {"vmean": 280.1187744140625, "vstd": 12.384744644165039},
            "VAR_SSTK_fine": {"vmean": 294.448974609375, "vstd": 7.419081211090088},
            "VAR_SKT_fine": {"vmean": 286.8110656738281, "vstd": 13.723491668701172},
            "VAR_ST_fine": {"vmean": 286.93658447265625, "vstd": 13.66016674041748},
            "VAR_TCWV_fine": {"vmean": 19.793996810913086, "vstd": 13.685070037841797},
            "VAR_2T_coarse": {"vmean": 286.2744445800781, "vstd": 12.475735664367676},
            "VAR_10U_coarse": {"vmean": 0.3791687488555908, "vstd": 3.261948585510254},
            "VAR_10V_coarse": {
                "vmean": -0.05645933374762535,
                "vstd": 3.4837522506713867,
            },
            "VAR_TP_coarse": {
                "vmean": 9.832954674493521e-05,
                "vstd": 0.0003440176951698959,
            },
            "VAR_D2M_coarse": {"vmean": 280.09478759765625, "vstd": 12.215740203857422},
            "VAR_SSTK_coarse": {"vmean": 292.5169982910156, "vstd": 8.795150756835938},
            "VAR_SKT_coarse": {"vmean": 286.7804260253906, "vstd": 13.49194049835205},
            "VAR_ST_coarse": {"vmean": 286.9049377441406, "vstd": 13.402721405029297},
            "VAR_TCWV_coarse": {"vmean": 19.75491714477539, "vstd": 13.40636920928955},
        }

        for key, values in RAW_CONSTANTS.items():
            norm_mapping[key] = EasyDict()
            norm_mapping[key].vmean = values["vmean"]
            norm_mapping[key].vstd = values["vstd"]

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


def coarse_down_up(
    fine_filtered, fine_batch, input_shape=(16, 32), axis=0, already_coarse=False
):
    """
    Downscale and then upscale fine-resolution data to compute coarse approximation.

    This function performs a downscaling-upscaling operation to create a coarse
    resolution approximation of fine data. This is commonly used in
    multi-scale analysis, image processing, and super-resolution tasks.

    Parameters
    ----------
    fine_filtered : torch.Tensor or np.ndarray
        Fine-resolution filtered data with shape (C, Hf, Wf) or (Hf, Wf).
        When `already_coarse=True`, it may instead contain native coarse data,
        such as CMIP6, which is directly upscaled to the fine resolution.
    fine_batch : torch.Tensor or np.ndarray
        Fine-resolution target data used to define the output spatial shape.
        Its spatial dimensions may differ from `fine_filtered` when
        `already_coarse=True`.
    input_shape : tuple of int, optional
        Target shape (Hc, Wc) for the coarse-resolution data after downscaling.
        Default is (16, 32).
    axis : int, optional
        Axis along which to insert batch dimension if the input lacks one.
        Default is 0.
    already_coarse : bool, optional
        If True, skip the downscaling step (the input is assumed to already be
        at the target coarse resolution `input_shape`) and only perform the
        upscaling back to the original fine resolution. Default is False.

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
    # out_shape = (fine_filtered.shape[-2], fine_filtered.shape[-1])
    # Final output must match the fine-grid resolution
    out_shape = (fine_batch.shape[-2], fine_batch.shape[-1])

    interp_transform = torchvision.transforms.Resize(
        out_shape,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )

    if already_coarse:
        # External data such as CMIP6 are already coarse: only upscale
        coarse_up = interp_transform(fine_filtered)
    else:
        # Standard case: downscale first, then upscale to the fine grid
        coarse_up = interp_transform(coarsen_transform(fine_filtered))

    # Remove batch dimension
    coarse_up = coarse_up.squeeze(0)

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


def make_starts(size, block, stride):
    """
    Generate block start indices covering an entire one-dimensional domain.

    The final block is adjusted, when necessary, so that it always reaches
    the domain boundary.

    Parameters
    ----------
    size : int
        Total size of the domain.
    block : int
        Size of each block.
    stride : int
        Distance between consecutive block starts.

    Returns
    -------
    list of int
        Start indices of the blocks.
    """

    starts = list(range(0, size - block + 1, stride))

    # Ensure the last block always touches the domain boundary,
    # regardless of whether the stride divides the domain evenly.
    last_start = size - block

    # The regular stride may stop before last_start. In that case, append an
    # additional block aligned with the end of the domain to avoid leaving an # uncovered region.

    # This final block may overlap more strongly with the previous block than
    # the blocks generated using the regular stride.
    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


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
        Target shape (height, width) for coarse resolution. Default is (80, 128).
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
        Run type: "train", "train_regional", "validation", "inference_regional", or "inference". Default is "train".
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
    region_center : tuple of float or None, optional
        Geographic center of the regional domain, expressed as
        "(latitude, longitude)" in degrees. Default is None.
    region_size : tuple of int or None, optional
        Spatial size of the regional domain. Default is None.
    overlap_ratio : float, optional
        Fractional overlap between adjacent spatial blocks during global
        inference. Default is 0.0.
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
    eval_region_indices : list or None
        Regional dataset key associated with each evaluation slice.
    train_slices : list of tuple
        Spatial slices used during regional training, when applicable.
    region_center : tuple of float or None
        Geographic center used for regional training or inference.
    region_size : tuple of int or None
        Size of the regional domain in latitude and longitude grid points.
    overlap_ratio : float
        Fractional overlap between adjacent blocks during global inference.
    random_centers : list of tuple or None
        List of random spatial centers for training mode.
    center_tracker : list
        Tracks spatial centers for debugging.
    tindex_tracker : list
        Tracks temporal indices for debugging.

    Notes
    -----
    - Supports both random (training) and deterministic (validation) sampling.
    - Handles cyclic longitude wrapping for global datasets.
    - Provides multi-scale processing through downscaling/upscaling.
    - Supports overlapping blocks during global inference for blended
      reconstruction.
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
        loaded_dfs_coarse=None,  # coarse input (CMIP6)
        in_shape=(80, 128),
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
        already_coarse=False,
        region_center=None,  # (lat_value, lon_value)
        region_size=None,
        region_constants_files=None,
        global_coordinates_file=None,
        coarse_resolution_deg=None,
        overlap_ratio=0.0,  # 0.02
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
        loaded_dfs_coarse : xarray.Dataset, optional
            External coarse dataset (CMIP6) used as coarse input.
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
        already_coarse : bool, optional
            Whether the external coarse dataset is already at in_shape. Default is False.
        region_center : tuple of float or None
            Fixed geographic center (lat, lon) for spatial sampling.
        region_size : tuple of int or None, optional
            Regional-domain size in latitude and longitude grid points.
        region_constants_files : list of str or None, optional
            Constant-variable files ordered like the regional datasets.
        global_coordinates_file : str or None, optional
            NetCDF file containing the global latitude and longitude coordinates.
        coarse_resolution_deg : float or None, optional
            Physical coarse-grid resolution in degrees.
        overlap_ratio : float, optional
            Fractional overlap between adjacent blocks during global inference.
        logger : logging.Logger, optional
            Logger instance.
        """
        self.constants_file_path = constants_file_path
        self.constant_variables = constant_variables
        self.years = years
        self.varnames_list = varnames_list
        self.units_list = units_list
        self.in_shape = in_shape
        assert len(self.in_shape) == 2, "in_shape must be a tuple of length 2"
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
        self.region_size = region_size
        self.region_constants_files = region_constants_files
        # Regional constant files follow the same order as the regional datasets.
        self.global_coordinates_file = global_coordinates_file
        self.global_latitudes = None
        self.global_longitudes = None
        self.coarse_resolution_deg = coarse_resolution_deg
        # A physical resolution lets each regional domain derive its own coarse shape.

        self.overlap_ratio = overlap_ratio

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

        self.already_coarse = already_coarse
        if self.already_coarse and loaded_dfs_coarse is None:
            raise ValueError(
                "already_coarse=True requires an external coarse dataset "
                "(loaded_dfs_coarse), e.g. CMIP6 already at coarse resolution."
            )

        if loaded_dfs_coarse is not None:
            if self.already_coarse:
                self.logger.info(
                    "External dataset will be used as source for coarse input. "
                    "It is already at coarse resolution, so only up-sampling will be applied."
                )
            else:
                self.logger.info(
                    "External dataset will be used as source for coarse input, "
                    "then down/up-sampled."
                )
        else:
            self.logger.info(
                "Fine dataset will be used as source for coarse input, "
                "then down/up-sampled."
            )

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

        self.region_const_vars = None
        # Keep regional constant arrays separate because regional shapes may differ.

        if self.constant_variables is not None and isinstance(loaded_dfs, dict):
            if self.region_constants_files is None:
                raise ValueError(
                    "region_constants_files is required when constant variables "
                    "are used in multi-region mode"
                )
            if len(self.region_constants_files) != len(loaded_dfs):
                raise ValueError(
                    "region_constants_files must contain one file per regional dataset"
                )

            self.const_vars = None
            self.region_const_vars = {}
            region_indices = list(loaded_dfs.keys())
            # Preserve the dictionary order established from region_datadirs.

            for region_index, constants_file in zip(
                region_indices, self.region_constants_files
            ):
                self.logger.info(
                    f"Opening constant variables file for region {region_index}: "
                    f"{constants_file}"
                )
                ds_const = xr.open_dataset(constants_file).load()

                if "time" in ds_const.dims:
                    ds_const = ds_const.isel(time=0)
                # Constant files may contain a singleton time dimension or no time.

                first_const = ds_const[self.constant_variables[0]]
                const_H, const_W = first_const.shape[-2:]
                region_const_vars = np.zeros(
                    (len(self.constant_variables), const_H, const_W)
                )
                # Allocate each regional array from its actual spatial dimensions.

                for i, const_varname in enumerate(self.constant_variables):
                    const_var = ds_const[const_varname]

                    if const_varname != "lsm":
                        self.logger.info(
                            f"Normalizing {const_varname} for region {region_index}"
                        )
                        latitude = (
                            ds_const.latitude
                            if hasattr(ds_const, "latitude")
                            else ds_const.lat
                        )
                        weighted_var = const_var.weighted(np.cos(np.radians(latitude)))
                        mean_var = weighted_var.mean().values
                        std_var = weighted_var.std().values
                        region_const_vars[i] = ((const_var - mean_var) / std_var).values
                    else:
                        region_const_vars[i] = const_var.values

                self.region_const_vars[region_index] = region_const_vars
                ds_const.close()
                # Store constants with the same key used by self.loaded_dfs.

            self.logger.info(
                f"Loaded regional constant variables: {self.constant_variables}"
            )
        elif (
            self.constant_variables is not None and self.constants_file_path is not None
        ):
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
        self.loaded_dfs_coarse = loaded_dfs_coarse
        self.multi_region = isinstance(loaded_dfs, dict)

        if self.multi_region:
            if isinstance(self.region_center, tuple):
                self.region_center = [self.region_center]
            if isinstance(self.region_size, tuple):
                self.region_size = [self.region_size]
            # Use the same internal list-of-tuples representation for one or many regions.

            if not loaded_dfs:
                raise ValueError("loaded_dfs cannot be empty")

            self.region_indices = list(loaded_dfs.keys())
            reference_dfs = loaded_dfs[self.region_indices[0]]

            for region_index in self.region_indices[1:]:
                region_time = loaded_dfs[region_index].time.values
                if not np.array_equal(region_time, reference_dfs.time.values):
                    raise ValueError(
                        "All regional datasets must have the same time coordinate"
                    )
            # A shared time index must refer to the same timestamp in every region.
        else:
            self.region_indices = None
            reference_dfs = loaded_dfs

        if self.global_coordinates_file is not None:
            self.logger.info(
                f"Opening global coordinates file: {self.global_coordinates_file}"
            )
            with xr.open_dataset(self.global_coordinates_file) as global_coordinates:
                if "latitude" in global_coordinates:
                    self.global_latitudes = global_coordinates["latitude"].values.copy()
                elif "lat" in global_coordinates:
                    self.global_latitudes = global_coordinates["lat"].values.copy()
                else:
                    raise ValueError(
                        "Global coordinates file must contain a 'latitude' or 'lat' coordinate."
                    )

                if "longitude" in global_coordinates:
                    self.global_longitudes = global_coordinates[
                        "longitude"
                    ].values.copy()
                elif "lon" in global_coordinates:
                    self.global_longitudes = global_coordinates["lon"].values.copy()
                else:
                    raise ValueError(
                        "Global coordinates file must contain a 'longitude' or 'lon' coordinate."
                    )

        self.valid_region_bounds = {}
        if (
            self.multi_region
            and self.global_coordinates_file is not None
            and self.region_size is not None
        ):
            global_lat = np.asarray(self.global_latitudes)
            global_lon = np.asarray(self.global_longitudes)
            half_lat = self.batch_size_lat // 2
            half_lon = self.batch_size_lon // 2

            for position, region_size in enumerate(self.region_size):
                region_index = self.region_indices[position]
                active_dfs = self.loaded_dfs[region_index]

                if hasattr(active_dfs, "latitude"):
                    region_latitudes = np.asarray(active_dfs.latitude.values)
                elif hasattr(active_dfs, "lat"):
                    region_latitudes = np.asarray(active_dfs.lat.values)
                else:
                    raise AttributeError(
                        "Regional dataset must have either 'latitude' or 'lat' coordinate."
                    )

                if hasattr(active_dfs, "longitude"):
                    region_longitudes = np.asarray(active_dfs.longitude.values)
                elif hasattr(active_dfs, "lon"):
                    region_longitudes = np.asarray(active_dfs.lon.values)
                else:
                    raise AttributeError(
                        "Regional dataset must have either 'longitude' or 'lon' coordinate."
                    )

                region_size_lat, region_size_lon = region_size
                if (
                    len(region_latitudes) != region_size_lat
                    or len(region_longitudes) != region_size_lon
                ):
                    raise ValueError(
                        f"Region {region_index} has data shape "
                        f"({len(region_latitudes)}, {len(region_longitudes)}) but "
                        f"region_size is ({region_size_lat}, {region_size_lon})"
                    )

                if (
                    region_size_lat < self.batch_size_lat
                    or region_size_lon < self.batch_size_lon
                ):
                    raise ValueError(
                        f"Region {region_index} with size {region_size} is smaller "
                        f"than batch size "
                        f"({self.batch_size_lat}, {self.batch_size_lon})"
                    )

                first_valid_lat = region_latitudes[half_lat]
                last_valid_lat = region_latitudes[region_size_lat - half_lat]
                first_valid_lat_index = int(
                    np.abs(global_lat - first_valid_lat).argmin()
                )
                last_valid_lat_index = int(np.abs(global_lat - last_valid_lat).argmin())

                first_valid_lon = region_longitudes[half_lon]
                last_valid_lon = region_longitudes[region_size_lon - half_lon]
                first_valid_lon_index = int(
                    np.abs(global_lon - first_valid_lon).argmin()
                )
                last_valid_lon_index = int(np.abs(global_lon - last_valid_lon).argmin())

                self.valid_region_bounds[region_index] = (
                    min(first_valid_lat_index, last_valid_lat_index),
                    max(first_valid_lat_index, last_valid_lat_index),
                    min(first_valid_lon_index, last_valid_lon_index),
                    max(first_valid_lon_index, last_valid_lon_index),
                )
                # Cache each region's valid global-center bounds once at initialization.

        self.time = reference_dfs.time
        self.etime = len(reference_dfs["time"])

        self.year = self.time.dt.year
        self.month = self.time.dt.month
        self.day = self.time.dt.day
        self.hour = self.time.dt.hour

        # ----------------------------------------------------------
        # 1. Extract time components from xarray
        # ----------------------------------------------------------

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
            # self.eval_slices = self.generate_evaluation_slices()
            # Overlapping blending only during inference.
            # overlap_ratio controls the trade-off between quality and compute cost:
            #   0.0 = hard-cut (no overlap)
            #   0.02 = 2% overlap, used here for lightweight blending
            #   0.1 to 0.2 = lightweight blending, ~30-75% more blocks
            #   0.5 = full Hann (perfect reconstruction property, ~3x more blocks)
            use_hann = self.run_type == "inference"

            if self.multi_region:
                if use_hann:
                    raise ValueError(
                        "Multi-region inference is not supported; provide a single "
                        "regional dataset for inference"
                    )

                self.eval_slices = []
                self.eval_region_indices = []
                # Keep historical four-value slices and track their regions separately.

                for region_index in self.region_indices:
                    region_dataset = self.loaded_dfs[region_index]
                    latitude_name = (
                        "latitude" if "latitude" in region_dataset.sizes else "lat"
                    )
                    longitude_name = (
                        "longitude" if "longitude" in region_dataset.sizes else "lon"
                    )
                    region_H = region_dataset.sizes[latitude_name]
                    region_W = region_dataset.sizes[longitude_name]

                    if region_H % self.batch_size_lat != 0:
                        raise ValueError(
                            f"Region {region_index} latitude size {region_H} must "
                            f"be divisible by batch_size_lat={self.batch_size_lat}"
                        )
                    if region_W % self.batch_size_lon != 0:
                        raise ValueError(
                            f"Region {region_index} longitude size {region_W} must "
                            f"be divisible by batch_size_lon={self.batch_size_lon}"
                        )
                    # Exact divisibility guarantees complete, non-overlapping coverage.

                    region_slices = self.generate_evaluation_slices(
                        use_hann_blending=False,
                        overlap_ratio=0.0,
                        domain_H=region_H,
                        domain_W=region_W,
                    )
                    self.eval_slices.extend(region_slices)
                    self.eval_region_indices.extend([region_index] * len(region_slices))
                    # Each local slice remains paired with its regional dataset key.
            else:
                self.eval_region_indices = None
                self.eval_slices = self.generate_evaluation_slices(
                    use_hann_blending=use_hann,
                    overlap_ratio=self.overlap_ratio,
                )
            # To Do: a key to add if all sbatch to taken or not
            self.sbatch = len(
                self.eval_slices
            )  # Set sbatch to exactly match number of slices
            self.logger.info(f"Evaluation mode: Generated {self.sbatch} spatial slices")
            # To Do: set time batches for validation (if inference is active or not)
            if self.run_type == "inference":
                self.time_batchs = np.arange(self.stime, self.etime, dtype=int)

            elif (
                self.run_type in ["inference_regional", "train_regional"]
            ):  # it can happen that we have mode = validation and run_type = train_regional.
                # Regional inference mode:
                # Instead of tiling the full spatial domain (as in global validation),
                # we extract a single spatial window centered on a user-defined
                # geographical location (latitude, longitude).

                # Ensure that a target region center is provided
                assert (
                    self.region_center is not None
                ), "region_center must be provided for inference_regional mode"

                assert (
                    self.region_size is not None
                ), "region_size must be provided when using inference_regional mode"

                lat_val, lon_val = self.region_center
                lat_idx, lon_idx = self.get_center_indices_from_latlon(lat_val, lon_val)

                region_size_lat, region_size_lon = self.region_size

                assert region_size_lat % self.batch_size_lat == 0
                assert region_size_lon % self.batch_size_lon == 0

                self.eval_slices = self.generate_region_slices(
                    lat_idx, lon_idx, region_size_lat, region_size_lon
                )

                # Set sbatch to exactly match number of slices
                self.sbatch = len(self.eval_slices)

                if self.run_type == "train_regional":
                    self.time_batchs = np.linspace(
                        self.etime // 3, self.etime * 2 // 3, self.tbatch, dtype=int
                    )
                else:
                    self.time_batchs = np.arange(self.stime, self.etime, dtype=int)

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
            if self.run_type == "train_regional":
                # Regional train mode:
                # Instead of randomly selecting windows (as in train),
                # we extract a single spatial window centered on a user-defined
                # geographical location (latitude, longitude).

                # Ensure that a target region center is provided
                assert (
                    self.region_center is not None
                ), "region_center must be provided for train_regional mode"

                assert (
                    self.region_size is not None
                ), "region_size must be provided when using train_regional mode"

                lat_val, lon_val = self.region_center
                lat_idx, lon_idx = self.get_center_indices_from_latlon(lat_val, lon_val)

                region_size_lat, region_size_lon = self.region_size

                assert region_size_lat % self.batch_size_lat == 0
                assert region_size_lon % self.batch_size_lon == 0

                self.train_slices = self.generate_region_slices(
                    lat_idx, lon_idx, region_size_lat, region_size_lon
                )

                # Set sbatch to exactly match number of slices
                self.sbatch = len(self.train_slices)
                self.time_batchs = np.arange(self.stime, self.etime, dtype=int)

                self.logger.info(
                    f"Train region mode activated at lat={lat_val}, lon={lon_val}"
                )
            else:  # train global
                self.time_batchs = np.arange(self.stime, self.etime, dtype=int)
                self.logger.info(
                    f"Training (global) mode: sbatch={self.sbatch}, tbatch={self.tbatch}"
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
        self.random_region_indices = [None] * self.sbatch if self.multi_region else None
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

    def get_center_indices_from_latlon(self, lat_value, lon_value, dataset=None):
        """
        Convert geographic coordinates (latitude, longitude) to nearest grid indices.

        Parameters
        ----------
        lat_value : float
            Latitude in degrees.
        lon_value : float
            Longitude in degrees.
        dataset : xarray.Dataset or None, optional
            Dataset whose latitude and longitude coordinates must be used.
            When omitted, the historical self.loaded_dfs dataset is used.

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

        active_dfs = self.loaded_dfs if dataset is None else dataset
        # Keep the historical dataset by default, or use the selected regional one.

        # Retrieve latitude and longitude arrays from the dataset
        if hasattr(active_dfs, "latitude"):
            lat_array = active_dfs.latitude.values
        elif hasattr(active_dfs, "lat"):
            lat_array = active_dfs.lat.values
        else:
            raise AttributeError(
                "Dataset must have either 'latitude' or 'lat' coordinate."
            )

        if hasattr(active_dfs, "longitude"):
            lon_array = active_dfs.longitude.values
        elif hasattr(active_dfs, "lon"):
            lon_array = active_dfs.lon.values
        else:
            raise AttributeError(
                "Dataset must have either 'longitude' or 'lon' coordinate."
            )

        # Find the index of the grid point closest to the requested lat/lon
        lat_idx = np.abs(lat_array - lat_value).argmin()
        lon_idx = np.abs(lon_array - lon_value).argmin()

        return lat_idx, lon_idx

    def valid_multiregional_centers(self, lat_random, lon_random):
        """Return the region containing a complete block around a global center."""
        for region_index, bounds in self.valid_region_bounds.items():
            valid_lat_start, valid_lat_end, valid_lon_start, valid_lon_end = bounds

            if (
                valid_lat_start <= lat_random <= valid_lat_end
                and valid_lon_start <= lon_random <= valid_lon_end
            ):
                return True, region_index

        return False, None

    def generate_random_batch_centers(self, n_batches, multi_regional=False):
        """
        Generate random (latitude, longitude) centers for batch sampling.

        Parameters
        ----------
        n_batches : int
            Number of random centers to generate.
        multi_regional : bool, optional
            Sample on the global grid and retain only centers contained in one
            of the configured regions.

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
        region_indices = []
        half_lat = self.batch_size_lat // 2

        if multi_regional:
            if self.global_latitudes is None or self.global_longitudes is None:
                raise ValueError(
                    "global_coordinates_file is required for multi-region sampling"
                )
            sampling_height = len(self.global_latitudes)
            sampling_width = len(self.global_longitudes)
        else:
            sampling_height = self.H
            sampling_width = self.W

        try:
            while len(centers) < n_batches:
                # Latitude: avoid poles (non-cyclic)
                lat_center = np.random.randint(half_lat, sampling_height - half_lat)
                # Longitude: any (cyclic)
                lon_center = np.random.randint(0, sampling_width)

                if multi_regional:
                    valid_center, region_index = self.valid_multiregional_centers(
                        lat_center, lon_center
                    )
                    if not valid_center:
                        continue
                    region_indices.append(region_index)

                centers.append((lat_center, lon_center))

            self.random_region_indices = region_indices if multi_regional else None

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

    def generate_evaluation_slices(
        self,
        use_hann_blending=False,
        overlap_ratio=0.0,
        domain_H=None,
        domain_W=None,
    ):
        """
        Generate deterministic spatial slices for evaluation mode.

        Parameters
        ----------
        use_hann_blending : bool, optional
            Whether to generate overlapping spatial blocks.
            This is intended for global inference only.
        overlap_ratio : float, optional
            Fraction of each block overlapping with adjacent blocks.
        domain_H : int or None, optional
            Height of the domain to tile. Defaults to self.H.
        domain_W : int or None, optional
            Width of the domain to tile. Defaults to self.W.

        Returns
        -------
        slices : list of tuple
            List of (lat_start, lat_end, lon_start, lon_end) tuples.
        """

        active_H = self.H if domain_H is None else domain_H
        active_W = self.W if domain_W is None else domain_W
        # Optional dimensions let the existing tiling logic serve each region.

        if not use_hann_blending:
            # Original validation behavior: keep it unchanged.
            n_blocks_lat = active_H // self.batch_size_lat
            n_blocks_lon = active_W // self.batch_size_lon

            # Create grid of block indices.
            lat_idx, lon_idx = np.mgrid[0:n_blocks_lat, 0:n_blocks_lon]

            # Calculate slice boundaries.
            lat_starts = (lat_idx * self.batch_size_lat).ravel()
            lon_starts = (lon_idx * self.batch_size_lon).ravel()

            lat_ends = lat_starts + self.batch_size_lat
            lon_ends = lon_starts + self.batch_size_lon

            # Create slices list.
            slices = list(zip(lat_starts, lat_ends, lon_starts, lon_ends))

            self.logger.info(
                f"Generated {len(slices)} evaluation blocks "
                f"({n_blocks_lat} x {n_blocks_lon} grid)"
            )

            return slices

        # Global inference behavior: overlapping blocks with full domain coverage.
        stride_lat = max(
            1,
            int(round(self.batch_size_lat * (1.0 - overlap_ratio))),
        )
        stride_lon = max(
            1,
            int(round(self.batch_size_lon * (1.0 - overlap_ratio))),
        )

        lat_starts = make_starts(
            active_H,
            self.batch_size_lat,
            stride_lat,
        )
        lon_starts = make_starts(
            active_W,
            self.batch_size_lon,
            stride_lon,
        )

        slices = []

        for lat_start in lat_starts:
            for lon_start in lon_starts:
                slices.append(
                    (
                        lat_start,
                        lat_start + self.batch_size_lat,
                        lon_start,
                        lon_start + self.batch_size_lon,
                    )
                )

        self.logger.info(
            f"Generated {len(slices)} overlapping evaluation blocks "
            f"({len(lat_starts)} x {len(lon_starts)} grid, "
            f"stride={stride_lat}x{stride_lon}, "
            f"overlap_ratio={overlap_ratio})"
        )

        return slices

    def generate_region_slices(
        self, lat_center, lon_center, region_size_lat, region_size_lon
    ):
        """
        Generate deterministic spatial slices for regional inference/train.
        The slices cover a block centered on (lat_center, lon_center).
        The region is divided into non-overlapping blocks of size
        (batch_size_lat, batch_size_lon) used for model inference.

        Parameters
        ----------
        lat_center : int
            Latitude index of the region center in the global grid.
        lon_center : int
            Longitude index of the region center in the global grid.
        region_size_lat : int
            Height of the region (in grid points).
        region_size_lon : int
            Width of the region (in grid points).

        Returns
        -------
        slices : list of tuple
            List of (lat_start, lat_end, lon_start, lon_end) tuples defining
            non-overlapping spatial blocks covering the selected region.

        Notes
        -----
        The latitude start index is clamped to ensure the region remains within
        the global latitude bounds. As a result, if the requested region is too
        close to the poles, the extracted region may be shifted and may no longer
        be centered exactly on (lat_center, lon_center). Longitude wrapping
        is handled later during patch extraction (in extract_batch).
        """

        n_blocks_lat = region_size_lat // self.batch_size_lat
        n_blocks_lon = region_size_lon // self.batch_size_lon

        # Compute the top-left corner (lat0, lon0) of the region
        # The region is centered around (lat_center, lon_center)
        lat0 = lat_center - region_size_lat // 2
        lon0 = lon_center - region_size_lon // 2

        # Latitude is NOT cyclic, so we must ensure indices stay within [0, H]
        lat0 = max(0, lat0)
        lat0 = min(lat0, self.H - region_size_lat)

        slices = []

        for i in range(n_blocks_lat):
            for j in range(n_blocks_lon):
                # Compute latitude boundaries of the current block
                lat_start = lat0 + i * self.batch_size_lat
                lat_end = lat_start + self.batch_size_lat

                # Compute longitude boundaries of the current block
                # Longitude is allowed to go out of bound
                lon_start = lon0 + j * self.batch_size_lon
                lon_end = lon_start + self.batch_size_lon

                slices.append((lat_start, lat_end, lon_start, lon_end))

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
            If the requested batch exceeds the input dimensions or if the
            latitude indices are invalid.

        Notes
        -----
        - Longitude is treated as cyclic (wraps around 0-360°).
        - Latitude is non-cyclic (no wrapping at poles).
        - The function rolls the data to center the longitude and then
          extracts the appropriate slice.
        """
        try:
            H, W = data.shape[-2:]
            # Regional datasets may have different full-domain spatial dimensions.

            assert (
                self.batch_size_lat <= H
            ), f"batch height {self.batch_size_lat} exceeds input height {H}"
            assert (
                self.batch_size_lon <= W
            ), f"batch width {self.batch_size_lon} exceeds input width {W}"

            half_lat = self.batch_size_lat // 2
            half_lon = self.batch_size_lon // 2

            # --- Compute latitude indices (non-cyclic) ---
            lat_start = ilat - half_lat
            lat_end = ilat + half_lat  # + 1  # +1 to include center line

            # --- Sanity check (should always hold given center generation logic) ---
            assert (
                0 <= lat_start <= H - self.batch_size_lat
            ), f"Invalid lat_start={lat_start}"
            assert self.batch_size_lat <= lat_end <= H, f"Invalid lat_end={lat_end}"

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

    def build_fine_coarse_blocks(
        self,
        npfeatures_full,
        npfeatures_coarse_source,
        lat_center,
        lon_center,
        input_shape=None,
    ):
        """
        Build fine, optionally filtered, and coarse-resolution spatial blocks
        centered at a given location.

        Parameters
        ----------
        npfeatures_full : np.ndarray
            Full-domain input features with shape (C, H, W).
        npfeatures_coarse_source : np.ndarray or None
            Optional external coarse-resolution features with shape (C, Hc, Wc).
            If None, the coarse input is derived from the fine-resolution data.
        lat_center : int
            Latitude center index.
        lon_center : int
            Longitude center index.
        input_shape : tuple of int or None, optional
            Coarse shape for the active domain. Falls back to self.in_shape.

        Returns
        -------
        fine_block : np.ndarray
            Extracted fine-resolution block of shape (C, batch_size_lat, batch_size_lon).
        fine_filtered_block : np.ndarray or None
            Filtered fine-resolution block of shape (C, batch_size_lat, batch_size_lon).
            Returns None if apply_filter is False.
        coarse_block : np.ndarray
            Coarse-resolution approximation of the fine block (after downscaling
            and upscaling).
        fine_indices : tuple
            Tuple of (lat_start, lat_end, lon_start, lon_end) indices defining the
            spatial region of the extracted blocks.

        Raises
        ------
        AssertionError
            If spatial indices mismatch between fine, filtered, and coarse blocks,
            or if their shapes are inconsistent.

        Notes
        -----
        - Spatial extraction, ensuring consistent handling of cyclic longitude
          and non-cyclic latitude.
        - External coarse data can be used directly as the coarse source.
        - The coarse block is generated from the full domain (not locally),
          ensuring global consistency of the low-resolution representation.
        - All returned blocks share identical spatial indices and shapes,
          making them directly comparable for residual learning.
        """
        fine_block, fine_indices = self.extract_batch(
            npfeatures_full, lat_center, lon_center
        )

        active_in_shape = self.in_shape if input_shape is None else input_shape
        # Keep the historical fixed shape unless an active regional shape is supplied.

        fine_filtered_block = None

        # Use external coarse data when available
        if npfeatures_coarse_source is not None:
            coarse_source_full = npfeatures_coarse_source

        else:
            # Otherwise derive the coarse source from the fine-resolution field
            if self.apply_filter:
                fine_filtered_full = self.filter_batch(npfeatures_full, fine_block)

                fine_filtered_block, filtered_indices = self.extract_batch(
                    fine_filtered_full,
                    lat_center,
                    lon_center,
                )

                assert filtered_indices == fine_indices, (
                    f"Indices mismatch after filtering:\n"
                    f"  fine indices: {fine_indices}\n"
                    f"  filtered indices: {filtered_indices}"
                )

                coarse_source_full = fine_filtered_full
            else:
                coarse_source_full = npfeatures_full

        # Build the full coarse field and resize it to the fine-grid resolution
        coarse_full = coarse_down_up(
            coarse_source_full,
            npfeatures_full,
            input_shape=active_in_shape,
            already_coarse=self.already_coarse,
        )

        coarse_block, coarse_indices = self.extract_batch(
            coarse_full,
            lat_center,
            lon_center,
        )

        assert coarse_indices == fine_indices, (
            f"Indices mismatch after coarsening:\n"
            f"  original indices: {fine_indices}\n"
            f"  coarse indices: {coarse_indices}"
        )

        assert coarse_block.shape == fine_block.shape, (
            f"Shape mismatch between fine and coarse blocks:\n"
            f"  fine shape: {fine_block.shape}\n"
            f"  coarse shape: {coarse_block.shape}"
        )

        return fine_block, fine_filtered_block, coarse_block, fine_indices

    def load_features(self, tindex, full_data_org):
        """
        Load fine features and, when available, external coarse features.
        External coarse data are currently supported only outside multi-regional mode.

        Parameters
        ----------
        tindex : int
            Time index to load.
        full_data_org : xarray.Dataset
            Fine dataset at the given time step (used for coordinates).

        Returns
        -------
        npfeatures_fine : np.ndarray
            Fine-resolution features of shape (C, H, W). Always ERA5.
        npfeatures_coarse_source : np.ndarray or None
            External coarse-resolution features of shape (C, Hc, Wc),
            or None when no external coarse dataset is used.
        """
        # Detect spatial dimension names from the fine dataset
        lat_dim = "latitude" if "latitude" in full_data_org.dims else "lat"
        lon_dim = "longitude" if "longitude" in full_data_org.dims else "lon"

        # Fine-grid spatial dimensions
        H = full_data_org.sizes[lat_dim]
        W = full_data_org.sizes[lon_dim]

        npfeatures_fine = np.zeros(
            [len(self.varnames_list), H, W],
            dtype=self.np_dtype,
        )
        # Fill each channel with the corresponding fine-resolution variable
        # The channel position is defined by index_mapping
        for var_name in self.varnames_list:
            iv = self.index_mapping[var_name]
            npfeatures_fine[iv] = full_data_org[var_name].values

        npfeatures_coarse_source = None

        # If an external coarse source is requested, load the matching time step
        # from the preloaded coarse dataset and store it using the same channel order
        if self.loaded_dfs_coarse is not None and not self.multi_region:
            full_data_coarse = self.loaded_dfs_coarse.isel(time=tindex)

            # Detect spatial dimension names from the external coarse dataset
            coarse_lat_dim = (
                "latitude" if "latitude" in full_data_coarse.dims else "lat"
            )
            coarse_lon_dim = (
                "longitude" if "longitude" in full_data_coarse.dims else "lon"
            )

            # External coarse-grid spatial dimensions
            Hc = full_data_coarse.sizes[coarse_lat_dim]
            Wc = full_data_coarse.sizes[coarse_lon_dim]

            npfeatures_coarse_source = np.zeros(
                [len(self.varnames_list), Hc, Wc], dtype=self.np_dtype
            )
            # Fill each channel with the corresponding extrenal coarse-resolution variable
            for var_name in self.varnames_list:
                iv = self.index_mapping[var_name]
                npfeatures_coarse_source[iv] = full_data_coarse[var_name].values

        return npfeatures_fine, npfeatures_coarse_source

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

        active_dfs = self.loaded_dfs
        active_const_vars = self.const_vars
        local_random_center = None
        # These defaults preserve the historical single-dataset execution path.

        if (
            self.multi_region
            and self.mode == "train"
            and self.run_type in ["train", "resume_train"]
        ):
            if self.last_tbatch_index != tbatch_index:
                self.random_centers = self.generate_random_batch_centers(
                    self.sbatch, multi_regional=True
                )
                self.last_tbatch_index = tbatch_index

            assert (
                self.random_centers[sindex] is not None
            ), f"Random center at index {sindex} has not been generated yet."
            assert (
                self.random_region_indices[sindex] is not None
            ), f"Region index at position {sindex} has not been generated yet."

            global_lat_center, global_lon_center = self.random_centers[sindex]
            region_index = self.random_region_indices[sindex]
            active_dfs = self.loaded_dfs[region_index]
            # The stored region index is the key of the selected regional dataset.
            active_const_vars = self.region_const_vars[region_index]
            # Select constants with the same key as the active weather dataset.

            lat_value = self.global_latitudes[global_lat_center]
            lon_value = self.global_longitudes[global_lon_center]
            # Convert global grid indices back to physical coordinates.
            local_random_center = self.get_center_indices_from_latlon(
                lat_value, lon_value, dataset=active_dfs
            )
            # The resulting center is expressed on the selected regional grid.

        elif self.multi_region and self.mode == "validation":
            region_index = self.eval_region_indices[sindex]
            active_dfs = self.loaded_dfs[region_index]
            active_const_vars = (
                self.region_const_vars[region_index]
                if self.region_const_vars is not None
                else None
            )
            # Validation slices select data and constants from the same region.

        # Load data
        full_data_org = active_dfs.isel(time=tindex)
        # From this point onward, processing uses a single xarray.Dataset.

        if hasattr(full_data_org, "latitude"):
            lat = full_data_org.latitude.values.copy()
        elif hasattr(full_data_org, "lat"):
            lat = full_data_org.lat.values.copy()
        else:
            raise AttributeError(
                f"Dataset must have either 'latitude' or 'lat' coordinate. "
                f"Available: {list(full_data_org.coords.keys())}"
            )

        if hasattr(full_data_org, "longitude"):
            lon = full_data_org.longitude.values.copy()
        elif hasattr(full_data_org, "lon"):
            lon = full_data_org.lon.values.copy()
        else:
            raise AttributeError(
                f"Dataset must have either 'longitude' or 'lon' coordinate. "
                f"Available: {list(full_data_org.coords.keys())}"
            )

        local_H = len(lat)
        local_W = len(lon)
        # Allocate intermediate arrays from the active region's actual dimensions.

        active_in_shape = self.in_shape
        if self.coarse_resolution_deg is not None:
            if local_H < 2 or local_W < 2:
                raise ValueError(
                    "At least two latitude and longitude points are required to "
                    "derive a coarse shape"
                )

            fine_resolution_lat = float(np.median(np.abs(np.diff(lat))))
            longitude_steps = np.abs((np.diff(lon) + 180.0) % 360.0 - 180.0)
            fine_resolution_lon = float(np.median(longitude_steps))
            # Derive the native grid spacing from the active region's coordinates.

            active_in_shape = (
                max(
                    1,
                    int(
                        round(
                            local_H * fine_resolution_lat / self.coarse_resolution_deg
                        )
                    ),
                ),
                max(
                    1,
                    int(
                        round(
                            local_W * fine_resolution_lon / self.coarse_resolution_deg
                        )
                    ),
                ),
            )
            # Different regional sizes now preserve the same physical coarse resolution.

        # Normalize to range [-1, 1] for better neural network input stability
        if self.global_latitudes is not None and self.global_longitudes is not None:
            lat_reference = self.global_latitudes
            lon_reference = self.global_longitudes
            # Use one shared geographic reference for all training regions.
        else:
            lat_reference = lat
            lon_reference = lon
            # Preserve the historical local normalization without a global file.

        lat_norm = (
            2
            * (
                (lat - lat_reference.min())
                / (lat_reference.max() - lat_reference.min())
            )
            - 1
        )
        lon_norm = (
            2
            * (
                (lon - lon_reference.min())
                / (lon_reference.max() - lon_reference.min())
            )
            - 1
        )
        # The same physical coordinate now receives the same value in every region.

        # 2D meshgrids of normalized latitude and longitude (shape: H x W)
        lat_grid, lon_grid = np.meshgrid(lat_norm, lon_norm, indexing="ij")

        sample = self.normalize_time(tindex)

        # Determine spatial sampling based on mode
        if self.mode == "validation":
            # Ensure evaluation slices are available
            assert (
                hasattr(self, "eval_slices") and self.eval_slices is not None
            ), "eval_slices not initialized for validation mode"
            assert len(self.eval_slices) > 0, "eval_slices is empty for validation mode"

            # Deterministic spatial sampling for evaluation.
            # eval_slices defines the spatial regions to evaluate and is later used
            # to reconstruct the full domain.
            lat_start, lat_end, lon_start, lon_end = self.eval_slices[sindex]

            lat_center = lat_start + self.batch_size_lat // 2
            lon_center = lon_start + self.batch_size_lon // 2

            self.center_tracker.append((lat_center, lon_center))

            if (
                self.run_type == "inference_regional"
                or self.run_type == "train_regional"
            ):
                # we can have mode = validation and run_type = train_regional at the same time.
                # Extract coordinates using the same spatial extraction logic as training
                # (cyclic longitude handling, boundary alignment).
                lat_batch, lat_indices = self.extract_batch(
                    lat_grid, lat_center, lon_center
                )
                lon_batch, lon_indices = self.extract_batch(
                    lon_grid, lat_center, lon_center
                )

                assert lat_indices == lon_indices, (
                    f"Indices mismatch for same center (lat_center={lat_center}, lon_center={lon_center}):\n"
                    f"  lat_indices: {lat_indices}\n"
                    f"  lon_indices: {lon_indices}"
                )

                lat_start, lat_end, lon_start, lon_end = lat_indices

                npfeatures_full, npfeatures_coarse_source = self.load_features(
                    tindex, full_data_org
                )

                fine_block, fine_filtered_block, coarse, fine_indices = (
                    self.build_fine_coarse_blocks(
                        npfeatures_full,
                        npfeatures_coarse_source,
                        lat_center,
                        lon_center,
                        input_shape=active_in_shape,
                    )
                )

            else:  # validation, global inference
                # Extract normalized coordinates for the batch
                lat_batch = lat_grid[lat_start:lat_end, lon_start:lon_end]
                lon_batch = lon_grid[lat_start:lat_end, lon_start:lon_end]

                # Extract data for all variables into the WHOLE spatial domain first (same as training)
                npfeatures_full, npfeatures_coarse_source = self.load_features(
                    tindex, full_data_org
                )

                # Extract the target batch for scaling determination
                fine_block = npfeatures_full[:, lat_start:lat_end, lon_start:lon_end]

                fine_filtered_block = None

                if npfeatures_coarse_source is not None:
                    coarse_source_full = npfeatures_coarse_source
                else:
                    if self.apply_filter:
                        fine_filtered_full = self.filter_batch(
                            npfeatures_full, fine_block
                        )

                        assert fine_filtered_full.shape == npfeatures_full.shape, (
                            f"Mismatch in shapes: fine_filtered has shape {fine_filtered_full.shape} "
                            f"but npfeatures_full has shape {npfeatures_full.shape}."
                        )

                        fine_filtered_block = fine_filtered_full[
                            :,
                            lat_start:lat_end,
                            lon_start:lon_end,
                        ]

                        assert fine_filtered_block.shape == fine_block.shape, (
                            f"Mismatch in shapes: fine_filtered_block has shape {fine_filtered_block.shape} "
                            f"but fine_block has shape {fine_block.shape}."
                        )

                        coarse_source_full = fine_filtered_full

                    else:
                        coarse_source_full = npfeatures_full

                coarse_full = coarse_down_up(
                    coarse_source_full,
                    npfeatures_full,
                    input_shape=active_in_shape,
                    already_coarse=self.already_coarse,
                )

                coarse = coarse_full[:, lat_start:lat_end, lon_start:lon_end]

                if self.debug:
                    self.logger.info(
                        f"  Evaluation mode: slice lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]\n"
                        f"  Full domain shape: {npfeatures_full.shape}\n"
                        f"  Batch shape: {fine_block.shape}"
                    )
        elif self.run_type == "train_regional":
            # Ensure evaluation slices are available
            assert (
                hasattr(self, "train_slices") and self.train_slices is not None
            ), "train_slices not initialized for train_regional mode"
            assert (
                len(self.train_slices) > 0
            ), "train_slices is empty for train_regional mode"

            # Deterministic spatial sampling for regional training.
            # train_slices defines the spatial regions to train on and is later used
            # to reconstruct the full domain.
            lat_start, lat_end, lon_start, lon_end = self.train_slices[sindex]

            lat_center = lat_start + self.batch_size_lat // 2
            lon_center = lon_start + self.batch_size_lon // 2

            self.center_tracker.append((lat_center, lon_center))
            # Extract coordinates using the same spatial extraction logic as training
            # (cyclic longitude handling, boundary alignment).
            lat_batch, lat_indices = self.extract_batch(
                lat_grid, lat_center, lon_center
            )
            lon_batch, lon_indices = self.extract_batch(
                lon_grid, lat_center, lon_center
            )

            assert lat_indices == lon_indices, (
                f"Indices mismatch for same center (lat_center={lat_center}, lon_center={lon_center}):\n"
                f"  lat_indices: {lat_indices}\n"
                f"  lon_indices: {lon_indices}"
            )

            lat_start, lat_end, lon_start, lon_end = lat_indices

            npfeatures_full, npfeatures_coarse_source = self.load_features(
                tindex, full_data_org
            )

            fine_block, fine_filtered_block, coarse, fine_indices = (
                self.build_fine_coarse_blocks(
                    npfeatures_full,
                    npfeatures_coarse_source,
                    lat_center,
                    lon_center,
                    input_shape=active_in_shape,
                )
            )

            assert fine_indices == lat_indices, (
                f"Indices mismatch between data and coordinate extractions:\n"
                f"  lat/lon indices: {lat_indices}\n"
                f"  data indices: {fine_indices}"
            )
        else:  # train (global)
            # Random spatial sampling for training
            if self.multi_region:
                assert (
                    local_random_center is not None
                ), "Local random center was not resolved for multi-region training."
                lat_center, lon_center = local_random_center
                # Extraction functions must receive indices on the local grid.
            else:
                if self.last_tbatch_index != tbatch_index:
                    self.random_centers = self.generate_random_batch_centers(
                        self.sbatch
                    )
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
            npfeatures_full, npfeatures_coarse_source = self.load_features(
                tindex, full_data_org
            )

            fine_block, fine_filtered_block, coarse, fine_indices = (
                self.build_fine_coarse_blocks(
                    npfeatures_full,
                    npfeatures_coarse_source,
                    lat_center,
                    lon_center,
                    input_shape=active_in_shape,
                )
            )

            assert fine_indices == lat_indices, (
                f"Indices mismatch between data and coordinate extractions:\n"
                f"  lat/lon indices: {lat_indices}\n"
                f"  data indices: {fine_indices}"
            )

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
            assert active_const_vars is not None, (
                f"Constant variables {self.constant_variables} were specified "
                f"but active_const_vars could not be loaded. Please check the file paths and variable names."
            )

            if self.mode == "validation":
                if (
                    self.run_type == "inference_regional"
                    or self.run_type == "train_regional"
                ):
                    # we can have mode validation and run_type train_regional at the same time.
                    # For inference_regional, use extract_batch
                    const_batch, const_indices = self.extract_batch(
                        active_const_vars, lat_center, lon_center
                    )
                    assert const_indices == lat_indices, (
                        f"Indices mismatch for constant variables:\n"
                        f"  coordinate indices: {lat_indices}\n"
                        f"  constant var indices: {const_indices}"
                    )

                else:  # validation, inference_global
                    # For evaluation, use direct slicing
                    const_batch = active_const_vars[
                        :, lat_start:lat_end, lon_start:lon_end
                    ]
            else:
                # For training (global or regional), use extract_batch
                const_batch, const_indices = self.extract_batch(
                    active_const_vars, lat_center, lon_center
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

        if self.run_type in ["inference_regional", "train_regional"]:
            # In regional inference / regional train mode, we extract a spatial window centered
            # on a user-defined (lat_center, lon_center)
            # Therefore, longitude requires special handling to avoid
            # discontinuities when crossing the dateline

            lat_vals = lat[lat_start:lat_end]

            # Recompute the same shift used inside extract_batch
            shift = self.W // 2 - lon_center
            lon_rolled = np.roll(lon, shift=shift)

            # Extract the regional longitude window
            lon_vals = lon_rolled[lon_start:lon_end]

            # Ensure longitude continuity when the window crosses the dateline
            if np.any(np.diff(lon_vals) < -180):
                lon_vals = np.rad2deg(np.unwrap(np.deg2rad(lon_vals)))

        else:
            lat_vals = lat[lat_start:lat_end]
            lon_vals = lon[lon_start:lon_end]

        sample.update(
            {
                "inputs": feature,  # model inputs (coarse_norm + coordinates + constants)
                "targets": residual,  # residual in normalized space (model target)
                "fine": fine_block,  # fine-resolution physical data (for diagnostics)
                "coarse": coarse,  # coarse physical data (baselinefor diagnostics)
                "corrdinates": {
                    "lat": torch.from_numpy(lat_vals).to(self.torch_dtype),
                    "lon": torch.from_numpy(lon_vals).to(self.torch_dtype),
                },
            }
        )

        if self.debug:
            self.logger.info("------------------------------------------------------")

        return sample
