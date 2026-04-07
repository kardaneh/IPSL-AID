# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Kazem Ardaneh, Kishanthan Kingston
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import sys
import time
import argparse
from IPSL_AID.logger import Logger
from IPSL_AID.utils import FileUtils, EasyDict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from IPSL_AID.dataset import stats, DataPreprocessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPSL_AID.model import load_model_and_loss
from IPSL_AID.model_utils import ModelUtils
import torch.optim as optim
import xarray as xr

from IPSL_AID.diagnostics import (
    plot_metric_histories,
    plot_loss_histories,
    plot_average_metrics,
    plot_spatiotemporal_histograms,
)

from IPSL_AID.evaluater import (
    MetricTracker,
    run_validation,
)


def parse_args():
    """
    Parse command line arguments for diffusion model training and inference.

    This function defines and parses all command line arguments required for
    configuring and running diffusion model training, resumption, or inference
    experiments. It provides comprehensive options for data loading, model
    architecture, training hyperparameters, and output management.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments as a namespace object with attributes
        corresponding to each argument.

    Notes
    -----
    - Arguments are organized into logical groups: execution mode, data
      configuration, training configuration, model architecture, and output.
    - Boolean arguments use string conversion with lambda functions for
      flexibility (accepts "true"/"false", "True"/"False", etc.).
    - Default values are provided for most parameters to allow minimal
      configuration for basic usage.
    - Some arguments have constraints or choices to ensure valid configurations.
    """
    parser = argparse.ArgumentParser(description="Train IPSL-AID diffusion model")

    # Execution mode and region
    parser.add_argument(
        "--debug",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable or disable debug mode",
    )

    parser.add_argument(
        "--region",
        type=lambda x: x.lower(),
        default=None,
        choices=["us", "europe", "asia"],
        help="region (only used if run_type=inference_regional)",
    )

    # Run configuration
    parser.add_argument(
        "--run_type",
        type=str,
        default="train",
        choices=["train", "resume_train", "inference", "inference_regional"],
        help="Run type: 'train', 'resume_train', 'inference' or 'inference_regional'",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="diffusion_model",
        help="Model name for inference (only for run_type=inference)",
    )

    parser.add_argument(
        "--inference_type",
        type=str,
        default="direct",
        choices=["direct", "sampler"],
        help="Inference mode: 'direct' for deterministic inference, 'sample' for stochastic sampling.",
    )

    # Data configuration
    parser.add_argument(
        "--varnames_list",
        type=str,
        nargs="+",
        default=["VAR_2T", "VAR_10U", "VAR_10V"],
        help="List of variable names to train on",
    )
    parser.add_argument(
        "--constant_varnames_list",
        type=str,
        nargs="+",
        default=["z", "lsm"],
        help="List of constant variable names",
    )
    parser.add_argument(
        "--constant_varnames_file",
        type=str,
        default="ERA5_const_sfc_variables.nc",
        help="Path to NetCDF file containing constant variables",
    )
    parser.add_argument(
        "--normalization_types",
        type=str,
        nargs="+",
        default=["VAR_2T=standard", "VAR_10U=standard", "VAR_10V=standard"],
        help="Normalization types for each variable as 'var=type' pairs",
    )
    parser.add_argument(
        "--dynamic_covariates",
        nargs="+",
        type=str,
        default=None,
        help="List of dynamic covariates",
    )
    parser.add_argument(
        "--dynamic_covariates_dir",
        type=str,
        default="../data_covariates/",
        help="Directory containing NetCDF files for dynamic covariates",
    )
    parser.add_argument(
        "--units_list",
        type=str,
        nargs="+",
        default=["K", "m/s", "m/s"],
        help="List of variable units corresponding to varnames_list",
    )

    # Time range configuration
    parser.add_argument(
        "--year_start", type=int, default=1980, help="Start year for dataset"
    )
    parser.add_argument(
        "--year_end", type=int, default=2020, help="End year for dataset"
    )
    parser.add_argument(
        "--year_start_test", type=int, default=2020, help="Start year for test dataset"
    )
    parser.add_argument(
        "--year_end_test", type=int, default=2022, help="End year for test dataset"
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )

    parser.add_argument(
        "--tbatch", type=int, default=1, help="Temporal batch length for processing"
    )

    parser.add_argument(
        "--sbatch",
        type=int,
        default=8,
        help="Number of spatial batches per timestamp for the traning",
    )

    parser.add_argument(
        "--train_temporal_batch_mode",
        type=str,
        default="partial",  # or "full"
        choices=["full", "partial"],
        help="Train temporal batch mode: 'full' for whole sequence, 'partial' for batched",
    )

    parser.add_argument(
        "--tbatch_train",
        type=int,
        default=1,
        help="Temporal batch length for training phase (only used when train_temporal_batch_mode='partial')",
    )

    parser.add_argument(
        "--test_temporal_batch_mode",
        type=str,
        default="full",  # or "different"
        choices=["full", "partial"],
        help="Test temporal batch mode: 'same' as training, 'different' for test-specific",
    )

    parser.add_argument(
        "--tbatch_test",
        type=int,
        default=None,
        help="Temporal batch length for test phase (only used when test_temporal_batch_mode='partial')",
    )

    parser.add_argument(
        "--test_spatial_batch_mode",
        type=str,
        default="full",  # or "partial"
        choices=["full", "partial"],
        help="Test spatial batch mode: 'full' for whole domain, 'partial' for batched processing",
    )

    parser.add_argument(
        "--sbatch_test",
        type=int,
        default=None,
        help="Number of spatial batches for test phase (only used when test_spatial_batch_mode=partial)",
    )

    parser.add_argument(
        "--batch_size_lat",
        type=int,
        default=145,
        help="Height of spatial batch in grid points (latitude direction), must be odd",
    )
    parser.add_argument(
        "--batch_size_lon",
        type=int,
        default=145,
        help="Width of spatial batch in grid points (longitude direction), must be odd",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--datadir", type=str, required=True, help="Dataset path")
    parser.add_argument(
        "--per_var_datadir",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )

    # Data processing parameters
    parser.add_argument(
        "--time_normalization",
        type=str,
        default="linear",
        help="Type of time normalization",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.02, help="Epsilon parameter for filtering"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Beta parameter for loss function"
    )
    parser.add_argument(
        "--margin", type=int, default=8, help="Margin parameter for filtering"
    )

    # Output configuration
    parser.add_argument(
        "--main_folder", type=str, default="experiment", help="Main output folder name"
    )
    parser.add_argument(
        "--sub_folder",
        type=str,
        default="experiment",
        help="Sub-folder name for current run",
    )
    parser.add_argument(
        "--prefix", type=str, default="run", help="Prefix for saved files"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "fp64"],
        help="Floating point precision",
    )

    # Diffusion model configuration
    parser.add_argument(
        "--arch",
        type=str,
        default="adm",
        choices=["ddpmpp", "ncsnpp", "adm"],
        help="Diffusion architecture type",
    )
    parser.add_argument(
        "--precond",
        type=str,
        default="edm",
        choices=["vp", "ve", "edm", "unet"],
        help="Diffusion preconditioner",
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Number of variable channels"
    )
    parser.add_argument(
        "--cond_channels", type=int, default=0, help="Number of conditioning channels"
    )
    parser.add_argument(
        "--out_channels", type=int, default=3, help="Number of output channels"
    )

    # Checkpoint configuration
    parser.add_argument(
        "--save_model",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Enable model checkpoint saving",
    )
    parser.add_argument(
        "--apply_filter",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Apply fine filtering for coarse data generation (default: True)",
    )
    parser.add_argument(
        "--save_checkpoint_name",
        type=str,
        default="diffusion_model_checkpoint",
        help="The name for saved checkpoints",
    )
    parser.add_argument(
        "--save_per_samples",
        type=int,
        default=10000,
        help="Save checkpoint every N samples",
    )

    parser.add_argument(
        "--load_checkpoint_name",
        type=str,
        default="model.pth.tar",
        help="Checkpoint file to load",
    )

    parser.add_argument(
        "--region_center",
        type=float,
        nargs=2,
        default=None,
        help="Latitude and longitude center for regional inference "
        "(used only when run_type=inference_regional)",
    )

    parser.add_argument(
        "--region_size",
        type=int,
        nargs=2,
        default=None,
        help="Requested regional size in grid points (lat lon) "
        "for regional inference (used only when run_type=inference_regional)",
    )

    # EDM sampler configuration
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of sampling steps used in the diffusion sampler",
    )

    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Minimum noise level used by the sampler",
    )

    parser.add_argument(
        "--sigma_max",
        type=float,
        default=80.0,
        help="Maximum noise level used by the sampler",
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=7.0,
        help="Exponent used for EDM time step discretization",
    )

    parser.add_argument(
        "--s_churn",
        type=float,
        default=40,
        help="Stochasticity strength parameter controlling noise injection during sampling",
    )

    parser.add_argument(
        "--s_min",
        type=float,
        default=0,
        help="Minimum noise level at which stochasticity is applied",
    )

    parser.add_argument(
        "--s_max",
        type=float,
        default=float("inf"),
        help="Maximum noise level at which stochasticity is applied",
    )

    parser.add_argument(
        "--s_noise",
        type=float,
        default=1.0,
        help="Noise scale applied when stochasticity is enabled",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="heun",
        choices=["heun", "euler"],
    )

    parser.add_argument(
        "--compute_crps", type=lambda x: x.lower() == "true", default=False
    )

    parser.add_argument("--crps_ensemble_size", type=int, default=10)
    parser.add_argument("--crps_batch_size", type=int, default=2)

    return parser.parse_args()


def make_divisible_hw(h, w, n):
    """
    Adjust height and width to be divisible by 2**n by decrementing.

    This function ensures that both the height (h) and width (w) are divisible
    by 2 raised to the power n, which is often required for neural network
    architectures that use pooling or strided convolutions multiple times.

    Parameters
    ----------
    h : int
        Original height value.
    w : int
        Original width value.
    n : int
        Exponent for divisor calculation. The divisor is 2**n.

    Returns
    -------
    h_new : int
        Adjusted height that is divisible by 2**n.
    w_new : int
        Adjusted width that is divisible by 2**n.

    Notes
    -----
    - The function decrements h and w until they become divisible by 2**n.
    - This is a common requirement for U-Net and other encoder-decoder architectures
      that use multiple downsampling and upsampling operations.
    - The adjustment is conservative (decrementing) to avoid adding padding,
      which might be important for maintaining exact spatial relationships.
    """
    div = 2**n

    # Fix H
    while h % div != 0:
        h -= 1

    # Fix W
    while w % div != 0:
        w -= 1

    return h, w


def setup_directories_and_logging(args):
    """
    Set up directory structure and logging infrastructure for experiments.

    This function creates a standardized directory hierarchy for organizing
    experiment outputs (logs, results, model checkpoints, etc.) and initializes
    a logging system with both console and file output.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Configuration object containing the following attributes:

        - main_folder : str
            Main experiment folder name.
        - sub_folder : str
            Sub-folder name for the current run.
        - prefix : str
            Prefix for log files and outputs.
        - datadir : str
            Base data directory path.
        - constant_varnames_file : str
            Filename for constant variables data.

    Returns
    -------
    paths : EasyDict
        Dictionary containing paths to created directories:

        - logs : str
            Path to log files directory.
        - results : str
            Path to results output directory.
        - runs : str
            Path to experiment run tracking directory.
        - checkpoints : str
            Path to model checkpoint directory.
        - stats : str
            Path to statistics and metrics directory.
        - datadir : str
            Original data directory path.
        - constants : str
            Full path to constant variables file.

    logger : Logger
        Configured logger instance with console and file output.

    Notes
    -----
    - Directory structure:

        logs/main_folder/sub_folder/
        results/main_folder/sub_folder/
        runs/main_folder/sub_folder/
        checkpoints/main_folder/sub_folder/
        stats/main_folder/sub_folder/

    - Log files are named with timestamp: {prefix}_log.txt
    - The logger outputs to both console and file by default.
    - All directories are created if they don't exist (via FileUtils.makedir).
    """
    # now = datetime.datetime.now()
    # date_time_str = now.strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(parent_dir)

    paths = EasyDict()
    paths.logs = os.path.join(project_root, "logs", args.main_folder, args.sub_folder)
    paths.results = os.path.join(
        project_root, "results", args.main_folder, args.sub_folder
    )
    paths.runs = os.path.join(project_root, "runs", args.main_folder, args.sub_folder)
    paths.checkpoints = os.path.join(
        project_root, "checkpoints", args.main_folder, args.sub_folder
    )
    paths.stats = os.path.join(project_root, "stats", args.main_folder, args.sub_folder)
    paths.stats_dir = os.path.join(project_root, "stats")
    paths.datadir = args.datadir
    paths.constants = os.path.join(paths.datadir, args.constant_varnames_file)

    # Create directories
    for path in [paths.logs, paths.results, paths.runs, paths.checkpoints, paths.stats]:
        FileUtils.makedir(path)

    # Setup logger
    log_file = os.path.join(paths.logs, f"{args.prefix}_log.txt")
    logger = Logger(
        console_output=True, file_output=True, log_file=log_file, record=True
    )
    logger.show_header("Main")

    return paths, logger


def log_configuration(args, paths, logger):
    """
    Log all configuration parameters to the provided logger.

    This function comprehensively logs all experiment configuration parameters
    including execution mode, data settings, training hyperparameters, model
    architecture, and directory structure. It provides a clear overview of the
    experiment setup for reproducibility and debugging.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Configuration object containing all experiment parameters.
    paths : EasyDict
        Dictionary containing paths to various experiment directories.
    logger : Logger
        Logger instance for outputting configuration information.

    Notes
    -----
    - The function organizes parameters into logical sections for readability.
    - Includes both user-specified parameters and derived directory paths.
    - Provides warnings for important configuration choices (e.g., disabled
      checkpoint saving).
    - The output is formatted with clear section headers and indentation.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION PARAMETERS")
    logger.info("=" * 60)

    # Execution mode and system
    logger.info("Execution Mode:")
    logger.info(f" └── Debug: {args.debug}")
    logger.info(f" └── Run type: '{args.run_type}'")
    logger.info(f" └── Inference type: '{args.inference_type}'")
    logger.info(f" └── Region: '{args.region}'")
    logger.info(f" └── Apply filter: {args.apply_filter}")

    # Checkpoint configuration
    logger.info("\nCheckpoint Configuration:")
    logger.info(f" └── Save model: {args.save_model}")
    logger.info(f" └── Save checkpoint name: '{args.save_checkpoint_name}'")
    logger.info(f" └── Load checkpoint name: '{args.load_checkpoint_name}'")
    logger.info(f" └── Save per samples: {args.save_per_samples}")
    if args.model_name:
        logger.info(f" └── Model name: '{args.model_name}'")
    else:
        logger.info(" └── Model name: Not specified")

    # Data configuration
    logger.info("\nData Configuration:")
    logger.info(f" └── Variable names: {args.varnames_list}")
    logger.info(f" └── Constant variables: {args.constant_varnames_list}")
    logger.info(f" └── Constant variables file: '{args.constant_varnames_file}'")
    logger.info(
        f" └── Dynamic covariates: {args.dynamic_covariates if args.dynamic_covariates else 'None'}"
    )
    logger.info(f" └── Dynamic covariates dir: '{args.dynamic_covariates_dir}'")
    logger.info(f" └── Units list: {args.units_list}")
    logger.info(f" └── Normalization types: {args.normalization_types}")
    logger.info(f" └── Data directory: '{args.datadir}'")

    # Time range configuration
    logger.info("\nTime Range Configuration:")
    logger.info(f" └── Training years: {args.year_start}-{args.year_end}")
    logger.info(f" └── Test years: {args.year_start_test}-{args.year_end_test}")
    logger.info(f" └── Time normalization: '{args.time_normalization}'")

    # Training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f" └── Number of epochs: {args.num_epochs}")
    logger.info(f" └── Batch size: {args.batch_size}")
    logger.info(f" └── Number of workers: {args.num_workers}")
    logger.info(f" └── Learning rate: {args.learning_rate}")
    logger.info(f" └── Dtype: '{args.dtype}'")

    # Spatial-temporal batching
    logger.info("\nSpatial-Temporal Batching:")
    logger.info(f" └── Spatial batches: {args.sbatch}")
    logger.info(f" └── Temporal time steps: {args.tbatch}")
    logger.info(f" └── Batch size (lat): {args.batch_size_lat} grid points")
    logger.info(f" └── Batch size (lon): {args.batch_size_lon} grid points")

    # Data processing parameters
    logger.info("\nData Processing Parameters:")
    logger.info(f" └── Epsilon: {args.epsilon}")
    logger.info(f" └── Beta: {args.beta}")
    logger.info(f" └── Margin: {args.margin}")

    # Model architecture
    logger.info("\nModel Architecture:")
    logger.info(f" └── Architecture: '{args.arch}'")
    logger.info(f" └── Preconditioner: '{args.precond}'")
    logger.info(f" └── Input channels: {args.in_channels}")
    logger.info(f" └── Conditioning channels: {args.cond_channels}")
    logger.info(f" └── Output channels: {args.out_channels}")

    # Sampler configuration
    logger.info("\nSampler Configuration:")
    logger.info(f" └── solver: {args.solver}")
    logger.info(f" └── num_steps: {args.num_steps}")
    logger.info(f" └── sigma_min: {args.sigma_min}")
    logger.info(f" └── sigma_max: {args.sigma_max}")
    logger.info(f" └── rho: {args.rho}")
    logger.info(f" └── s_churn: {args.s_churn}")
    logger.info(f" └── s_min: {args.s_min}")
    logger.info(f" └── s_max: {args.s_max}")
    logger.info(f" └── s_noise: {args.s_noise}")

    # Output configuration
    logger.info("\nOutput Configuration:")
    logger.info(f" └── Main folder: '{args.main_folder}'")
    logger.info(f" └── Sub folder: '{args.sub_folder}'")
    logger.info(f" └── Prefix: '{args.prefix}'")

    # Directory paths (for reference)
    logger.info("\nGenerated Directory Paths:")
    logger.info(f" └── Logs directory: '{paths.logs}'")
    logger.info(f" └── Results directory: '{paths.results}'")
    logger.info(f" └── TensorBoard runs: '{paths.runs}'")
    logger.info(f" └── Model checkpoints: '{paths.checkpoints}'")
    logger.info(f" └── Statistics: '{paths.stats}'")
    logger.info(f" └── Statistics: '{paths.stats_dir}'")
    logger.info(f" └── Data directory: '{paths.datadir}'")
    logger.info(f" └── Constants file: '{paths.constants}'")

    # Per-variable data directories
    logger.info("\nPer-variable data directories:")
    if args.per_var_datadir is None:
        logger.info(" └── Using default data directory for all variables")
    else:
        for item in args.per_var_datadir:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")

    # Special notes based on run type
    logger.info("\nRun Type Notes:")
    if args.run_type == "train":
        logger.info(" └── Mode: Training from scratch")
    elif args.run_type == "resume_train":
        logger.info(" └── Mode: Resuming training from checkpoint")
    elif args.run_type == "inference":
        logger.info(" └── Mode: Inference only")
        logger.info(f" └── Inference type: {args.inference_type}")
        logger.info(f" └── Compute CRPS: {args.compute_crps}")
    elif args.run_type == "inference_regional":
        logger.info(" └── Mode: Regional inference")
        logger.info(f" └── Region center: {args.region_center}")
        logger.info(f" └── Region size: {args.region_size}")
        logger.info(f" └── Compute CRPS: {args.compute_crps}")

    # Checkpoint saving strategy
    if args.save_model:
        logger.info("\nCheckpoint Saving Strategy:")
        logger.info(" └── Checkpoints enabled:")
        logger.info(f" └── Saving every {args.save_per_samples:,} samples")
        logger.info(" └── Saving epoch checkpoints every 10 epochs")
        logger.info(" └── Saving best model based on validation MAE")
        logger.info(" └── Saving final model at end of training")
        logger.info(" └── Each checkpoint includes:")
        logger.info("     └── Model state dict")
        logger.info("     └── Optimizer state")
        logger.info("     └── Training/validation history")
        logger.info("     └── Metrics history")
        logger.info("     └── Training state (epoch, samples processed)")
        logger.info("     └── Configuration arguments")
    else:
        logger.info("\nCheckpoint Saving:")
        logger.info(" └── DISABLED - No checkpoints will be saved!")
        logger.info(
            "     └── Warning: Training progress cannot be resumed if interrupted"
        )

    logger.info("=" * 60)


def setup_data_paths(args, paths, logger):
    """
    Set up data file paths, load datasets, and compute normalization statistics.

    This function handles the data loading pipeline for training and validation
    datasets. It manages per-variable data paths, concatenates multi-year data
    for each variable, computes normalization statistics, and sets up variable
    mappings and normalization types.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Configuration object containing runtime options such as training years,
        execution mode, variable names, and normalization specifications.
    paths : EasyDict
        Dictionary containing directory paths.

        Expected keys:
        - datadir
        - stats
    logger : logging.Logger
        Logger instance for output messages.

    Returns
    -------
    norm_mapping : dict
        Mapping from variable name to normalization statistics.
    steps : EasyDict
        Grid dimension information (time, latitude, longitude).
    normalization_type : EasyDict
        Mapping from variable name to normalization method.
    index_mapping : dict
        Mapping from variable name to array index.
    train_ds : xarray.Dataset or None
        Training dataset, or None if run_type is ``inference``.
    valid_ds : xarray.Dataset
        Validation dataset.

    Notes
    -----
    - Per-variable data directories may be provided using ``VAR=path`` syntax.
    - Training data is only loaded when ``run_type`` is not ``inference``.
    - Normalization statistics are computed on the validation dataset.
    - Variables from different files and years are merged into a single dataset.
    """
    train_years = np.arange(args.year_start, args.year_end + 1)
    test_years = np.arange(args.year_start_test, args.year_end_test + 1)
    logger.info(f"Training years: {train_years}")
    logger.info(f"Testing years: {test_years}")

    # ------------------------------------------------------------------
    # Per-variable data paths configuration (using EasyDict)
    # ------------------------------------------------------------------
    # Default path is used as a fallback when a variable-specific path
    # is not provided via the command line.
    per_var_paths = EasyDict()
    per_var_paths.default = paths.datadir
    # logger.info(f"[DEBUG] args.per_var_datadir = {args.per_var_datadir}")

    # Per-variable data directories passed as VAR=path
    if args.per_var_datadir is not None:
        for item in args.per_var_datadir:
            var, path = item.split("=")
            per_var_paths[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")

    logger.info(f"[Data paths] default → {per_var_paths.default}")

    # --------------------------
    # Training datasets
    # --------------------------
    train_ds = None

    # if args.run_type != "inference":
    if args.run_type not in ["inference", "inference_regional"]:
        logger.info("Pre-loading training datasets...")

        train_var_datasets = []

        # Load each variable independently, then concatenate along time
        for var in args.varnames_list:
            base_path = per_var_paths.get(var, per_var_paths.default)

            train_filenames = [f"{base_path}/samples_{year}.nc" for year in train_years]

            logger.info(
                f"{var} training files:\n[\n"
                + "\n".join(f"  {f}" for f in train_filenames)
                + "\n]"
            )

            # Open first dataset and sort by time
            ds_var = xr.open_dataset(train_filenames[0]).sortby("time")

            # Loop through remaining files and concatenate along time
            for fname in train_filenames[1:]:
                ds_next = xr.open_dataset(fname).sortby("time")
                ds_var = xr.concat([ds_var, ds_next], dim="time")

            # Keep only the current variable before merging
            train_var_datasets.append(ds_var[[var]])

        # Merge all variables into a single training dataset
        train_ds = xr.merge(train_var_datasets).load()
        logger.info(f"Training dataset concatenated: {train_ds.sizes}")

    else:
        logger.info("Inference mode: skipping training dataset loading")

    # --------------------------
    # Validation datasets
    # --------------------------
    logger.info("Pre-loading validation datasets...")

    valid_var_datasets = []

    # Load each variable independently, then concatenate along time
    for var in args.varnames_list:
        base_path = per_var_paths.get(var, per_var_paths.default)

        valid_filenames = [f"{base_path}/samples_{year}.nc" for year in test_years]

        logger.info(
            f"{var} validation files:\n[\n"
            + "\n".join(f"  {f}" for f in valid_filenames)
            + "\n]"
        )

        # Open first validation dataset
        ds_var = xr.open_dataset(valid_filenames[0]).sortby("time")

        # Loop through remaining validation files
        for fname in valid_filenames[1:]:
            ds_next = xr.open_dataset(fname).sortby("time")
            ds_var = xr.concat([ds_var, ds_next], dim="time")

        # Keep only the current variable before merging
        valid_var_datasets.append(ds_var[[var]])

    # Merge all variables into a single validation dataset
    valid_ds = xr.merge(valid_var_datasets).load()
    logger.info(f"Validation dataset concatenated: {valid_ds.sizes}")

    # norm_mapping, steps = stats(train_ds, logger, paths.stats)
    norm_mapping, steps = stats(valid_ds, logger, paths.stats_dir)
    assert hasattr(steps, "time"), "steps does not contain a 'time' attribute"

    # Setup normalization types
    normalization_type = EasyDict()
    for mapping in args.normalization_types:
        if "=" in mapping:
            var_name, norm_type = mapping.split("=")
            normalization_type[var_name] = norm_type
        else:
            logger.warning(
                f"Invalid normalization mapping: {mapping}. Expected 'VAR_NAME=type'"
            )

    # Verify all variables have normalization types
    for var in args.varnames_list:
        if var not in normalization_type:
            logger.warning(
                f"Variable '{var}' not found in normalization_types. Defaulting to 'standard'"
            )
            normalization_type[var] = "standard"

    logger.info(f"Normalization types: {normalization_type}")

    # Create index mapping
    index_mapping = {var_name: i for i, var_name in enumerate(args.varnames_list)}
    for var_name, idx in index_mapping.items():
        logger.info(f"{var_name}: Index {idx}")

    # Log normalization statistics
    logger.info("------ Normalization Statistics (norm_mapping) ------")
    for key, st in norm_mapping.items():
        logger.info(
            f"\n[{key}]\n"
            f" └── vmin={getattr(st, 'vmin', None)}\n"
            f" └── vmax={getattr(st, 'vmax', None)}\n"
            f" └── vmean={getattr(st, 'vmean', None)}\n"
            f" └── vstd={getattr(st, 'vstd', None)}\n"
            f" └── median={getattr(st, 'median', None)}\n"
            f" └── iqr={getattr(st, 'iqr', None)}\n"
            f" └── q1={getattr(st, 'q1', None)}\n"
            f" └── q3={getattr(st, 'q3', None)}"
        )
    logger.info("------------------------------------------------------")

    return norm_mapping, steps, normalization_type, index_mapping, train_ds, valid_ds


def setup_training_environment(args, logger):
    """
    Set up the training environment including device selection, random seeds,
    and data type configuration.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Configuration object containing precision settings.
    logger : logging.Logger
        Logger instance for output messages.

    Returns
    -------
    device : torch.device
        Selected computing device.
    torch_dtype : torch.dtype
        PyTorch data type.
    np_dtype : numpy.dtype
        NumPy data type.
    use_fp16 : bool
        Whether half precision is enabled.

    Notes
    -----
    - Sets global random seeds for reproducibility.
    - Automatically selects CUDA if available.
    - Enables PyTorch anomaly detection for debugging.
    """
    # Set random seeds for reproducibility
    random_state = 0
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.set_printoptions(precision=5)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Setup data types
    torch_dtype_map = EasyDict(
        {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}
    )
    np_dtype_map = EasyDict(
        {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}
    )

    torch_dtype = torch_dtype_map[args.dtype]
    np_dtype = np_dtype_map[args.dtype]
    use_fp16 = torch_dtype == torch.float16

    return device, torch_dtype, np_dtype, use_fp16


def create_data_loaders(
    args,
    paths,
    norm_mapping,
    steps,
    normalization_type,
    index_mapping,
    torch_dtype,
    np_dtype,
    logger,
    mode="train",
    run_type="train",
    train_loaded_dfs=None,
    valid_loaded_dfs=None,
):
    """
    Create data loaders for training, validation, or inference.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Runtime configuration options.
    paths : EasyDict
        Directory paths including constants files.
    norm_mapping : dict
        Normalization statistics per variable.
    steps : EasyDict
        Grid dimension information.
    normalization_type : EasyDict
        Normalization method per variable.
    index_mapping : dict
        Variable-to-index mapping.
    torch_dtype : torch.dtype
        PyTorch tensor dtype.
    np_dtype : numpy.dtype
        NumPy array dtype.
    logger : logging.Logger
        Logger instance.
    mode : str, optional
        Either ``train`` or ``validation``.
    run_type : str, optional
        Execution mode (train, resume_train, inference).
    train_loaded_dfs : dict, optional
        Pre-loaded training datasets.
    valid_loaded_dfs : dict, optional
        Pre-loaded validation datasets.

    Returns
    -------
    data_loader : torch.utils.data.DataLoader
        Configured data loader.
    img_res : tuple of int
        Spatial resolution used by the model.
    dataset : DataPreprocessor
        Underlying dataset object.

    Raises
    ------
    ValueError
        If ``mode`` is invalid.
    AssertionError
        If required datasets are missing.

    Notes
    -----
    - Spatial dimensions are adjusted to be divisible by powers of two.
    - Validation falls back to training data if validation data is unavailable.
    - Data is assumed to be pre-loaded into memory.
    """
    # Validate mode parameter
    if mode not in ["train", "validation"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'validation'")

    # Adjust resolution for model compatibility
    n = args.depth if hasattr(args, "Unet_depth") else 3
    h, w = make_divisible_hw(args.batch_size_lat, args.batch_size_lon, n)
    img_res = (h, w)

    logger.info(f"Creating {mode} dataset:")
    logger.info(
        f" └── Original resolution: ({args.batch_size_lat}, {args.batch_size_lon})"
    )
    logger.info(f" └── Adjusted to divisible-by-2^{n} resolution: {img_res}")

    # Determine dataset parameters based on mode with assertions
    if mode == "train":
        years = np.arange(args.year_start, args.year_end + 1)
        assert (
            train_loaded_dfs is not None
        ), "train_loaded_dfs must be provided for training mode"
        assert len(train_loaded_dfs) > 0, "train_loaded_dfs must not be empty"
        loaded_dfs = train_loaded_dfs
        shuffle = True  # Shuffle for training
        tbatch = args.tbatch
        sbatch = args.sbatch
    else:  # validation
        years = np.arange(args.year_start_test, args.year_end_test + 1)
        if valid_loaded_dfs is None or len(valid_loaded_dfs) == 0:
            logger.warning(
                "No validation data provided, using training data for validation"
            )
            assert (
                train_loaded_dfs is not None
            ), "train_loaded_dfs must be provided as fallback for validation"
            assert len(train_loaded_dfs) > 0, "train_loaded_dfs must not be empty"
            loaded_dfs = train_loaded_dfs
            years = np.arange(
                args.year_start, args.year_start + 1
            )  # Use first training year
        else:
            loaded_dfs = valid_loaded_dfs
        shuffle = False  # No shuffle for validation
        # Use smaller batches for validation to save memory
        tbatch = args.batch_size  # same as torch batch size
        sbatch = args.sbatch  # Half the spatial batches

    logger.info(f" └── {mode} years: {years}")
    logger.info(
        f" └── {mode} parameters - tbatch: {tbatch}, sbatch: {sbatch}, shuffle: {shuffle}"
    )
    logger.info(f" └── Number of {mode} files: {len(years)}")

    # Create dataset with pre-loaded data
    dataset = DataPreprocessor(
        years=years,  # List of years
        loaded_dfs=loaded_dfs,  # Pre-loaded datasets dictionary
        constants_file_path=paths.constants,
        varnames_list=args.varnames_list,
        units_list=args.units_list,
        in_shape=(80, 128),
        batch_size_lat=h,
        batch_size_lon=w,
        steps=steps,
        tbatch=tbatch,
        sbatch=sbatch,
        debug=args.debug,
        mode=mode,
        run_type=run_type,
        dynamic_covariates=args.dynamic_covariates,
        dynamic_covariates_dir=args.dynamic_covariates_dir,
        time_normalization=args.time_normalization,
        norm_mapping=norm_mapping,  # Same normalization for consistency
        index_mapping=index_mapping,
        normalization_type=normalization_type,
        constant_variables=args.constant_varnames_list,
        epsilon=args.epsilon,
        margin=args.margin,
        dtype=(torch_dtype, np_dtype),  # Same dtype for consistency
        apply_filter=args.apply_filter,
        region_center=args.region_center,
        region_size=args.region_size,
        logger=logger,
    )

    # Create data loader - set num_workers=0 since data is pre-loaded
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,  # Data is pre-loaded, no workers needed
        pin_memory=True,
    )

    logger.info(f"  {mode} dataset size: {len(dataset)}")
    logger.info(f"  {mode} data loader batches: {len(data_loader)}")

    return data_loader, img_res, dataset


def setup_model(args, img_res, use_fp16, device, logger):
    """
    Set up the diffusion model and its loss function.

    Parameters
    ----------
    args : argparse.Namespace or EasyDict
        Model configuration options.
    img_res : tuple of int
        Image resolution (height, width).
    use_fp16 : bool
        Whether FP16 precision is enabled.
    device : torch.device
        Target device.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    model : torch.nn.Module
        Initialized model.
    loss_fn : callable
        Loss function.

    Raises
    ------
    ValueError
        If an unsupported time normalization is specified.

    Notes
    -----
    - Label dimensionality depends on the selected time normalization.
    - Model creation is delegated to ``load_model_and_loss``.
    """
    logger.info("------ Model and Loss info ------")

    # Determine label_dim based on time_normalization
    if args.time_normalization == "linear":
        label_dim = 2
    elif args.time_normalization == "cos_sin":
        label_dim = 4
    else:
        raise ValueError(f"Unsupported time_normalization: {args.time_normalization}")

    logger.info(
        f"Label dimension: {label_dim} (time_normalization: {args.time_normalization})"
    )

    opts = EasyDict(
        {
            "arch": args.arch,
            "precond": args.precond,
            "img_resolution": img_res,
            "in_channels": args.in_channels,
            "cond_channels": args.cond_channels,
            "out_channels": args.out_channels,
            "label_dim": label_dim,
            "use_fp16": use_fp16,
        }
    )

    model, loss_fn = load_model_and_loss(opts, logger=logger, device=device)

    return model, loss_fn


def resolve_region_center(args):
    """
    Resolve the regional inference center coordinates.

    This function enforces the logic for regional inference:
    user can provide either region or region_center

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    tuple or None
        (lat, lon) if inference_regional,
        None otherwise.
    Raises
    ------
    ValueError
        If both region and region_center are provided,
        neither is provided in inference_regional mode,
        an unknown region name is specified.

    Notes
    -----
    - Predefined regions map to fixed center coordinates.
    - Longitude follows the convention [0, 360].
    """

    if args.run_type != "inference_regional":
        return None

    # predefined region center coordinates (lat, lon)
    # longitude convention: [0, 360]
    region_coords = {
        "us": (39.0, 262.0),
        "europe": (50.0, 10.0),
        "asia": (35.0, 100.0),
    }

    # case 1: both provided (error)
    if args.region is not None and args.region_center is not None:
        raise ValueError("provide either --region or --region_center, not both.")

    # case 2: predefined region
    if args.region is not None:
        if args.region not in region_coords:
            raise ValueError(
                f"invalid region '{args.region}'. "
                f"available regions: {list(region_coords.keys())}"
            )

        return region_coords[args.region]

    # case 3: explicit coordinates
    if args.region_center is not None:
        if len(args.region_center) != 2:
            raise ValueError("--region_center must contain exactly two values: lat lon")
        return tuple(args.region_center)

    # case 4: nothing provided (error)
    raise ValueError(
        "for run_type='inference_regional', you must provide "
        "either --region (us, europe, asia) or --region_center lat lon."
    )


def main():
    """
    Main training and inference pipeline for IPSL-AID diffusion models.

    This function orchestrates the entire training and inference process for
    diffusion-based generative models on weather and climate data. It handles:
    - Argument parsing and configuration
    - Directory setup and logging
    - Data loading and preprocessing
    - Model initialization and checkpoint management
    - Training loop with validation
    - Inference execution
    - Visualization and result saving

    The pipeline supports multiple modes of operation:
    - Training from scratch (run_type='train')
    - Resuming training from a checkpoint (run_type='resume_train')
    - Running inference with a trained model (run_type='inference')

    The function follows a structured workflow:
    1. Parse command line arguments
    2. Setup directories and logging
    3. Load and preprocess data
    4. Initialize model, optimizer, and loss function
    5. Handle checkpoint loading if required
    6. Execute training loop with validation or run inference
    7. Generate plots and save results

    Parameters
    ----------
    None
        All configuration is provided via command line arguments.

    Returns
    -------
    None

    Notes
    -----
    - The function uses argparse for command line argument parsing.
    - All output (logs, checkpoints, results) is saved to organized directories.
    - Training includes validation at each epoch with metrics tracking.
    - Inference mode runs validation metrics without training.
    - Mixed precision training (FP16) is supported when available.
    - Model checkpoints include full training state for resumption.
    - TensorBoard integration is provided for training visualization.

    Raises
    ------
    FileNotFoundError
        If required checkpoints are not found for resumption or inference.
    RuntimeError
        If inference mode is requested without validation data.
    ValueError
        If invalid configurations are provided.
    """
    # Check for --version flag
    if "--version" in sys.argv or "-V" in sys.argv:
        from IPSL_AID import __version__, __author__, __license__

        print(f"IPSL-AID version {__version__}")
        print(f"Copyright (c) 2026 {__author__}")
        print(f"License: {__license__}")
        print("Repository: https://github.com/kardaneh/IPSL-AID")
        sys.exit(0)

    # Parse command line arguments
    args = parse_args()

    args.region_center = resolve_region_center(args)

    # Setup directories and logging
    paths, logger = setup_directories_and_logging(args)

    # Log configuration parameters
    log_configuration(args, paths, logger)

    # Setup data paths and normalization statistics
    (
        norm_mapping,
        steps,
        normalization_type,
        index_mapping,
        train_loaded_dfs,
        valid_loaded_dfs,
    ) = setup_data_paths(args, paths, logger)

    # Setup training environment (device, data types, random seeds)
    device, torch_dtype, np_dtype, use_fp16 = setup_training_environment(args, logger)

    # Setup TensorBoard for visualization
    # if args.run_type != "inference":
    if args.run_type not in ["inference", "inference_regional"]:
        writer = SummaryWriter(f"runs/{args.main_folder}/{args.sub_folder}/")
        logger.info(
            f"TensorBoard enabled at: runs/{args.main_folder}/{args.sub_folder}/"
        )
    else:
        writer = None
        logger.info("TensorBoard disabled for inference mode")

    # Create data loaders
    # if args.run_type != "inference":
    if args.run_type not in ["inference", "inference_regional"]:
        train_loader, img_res, train_dataset = create_data_loaders(
            args,
            paths,
            norm_mapping,
            steps,
            normalization_type,
            index_mapping,
            torch_dtype,
            np_dtype,
            logger,
            mode="train",
            train_loaded_dfs=train_loaded_dfs,
        )
        logger.info(f"Training dataset loaded with image resolution: {img_res}")
    else:
        logger.info("Inference mode: Skipping training data loader creation")
        train_loader, img_res, train_dataset = None, None, None

    if valid_loaded_dfs is not None:
        valid_loader, valid_img_res, valid_dataset = create_data_loaders(
            args,
            paths,
            norm_mapping,
            steps,
            normalization_type,
            index_mapping,
            torch_dtype,
            np_dtype,
            logger,
            mode="validation",
            run_type=args.run_type,
            valid_loaded_dfs=valid_loaded_dfs,
        )
        logger.info(f"Validation dataset loaded with image resolution: {valid_img_res}")
        # if args.run_type == "inference":
        if args.run_type in ["inference", "inference_regional"]:
            img_res = valid_img_res  # Use validation image resolution for inference
    else:
        valid_loader, valid_img_res, valid_dataset = None, None, None
        logger.warning("No validation dataset created (test files not found or empty)")

    if args.run_type == "inference" and valid_loader is None:
        logger.error(
            "Inference mode requires a validation dataset, but none was created."
        )
        raise RuntimeError("Cannot run inference without a validation dataset.")

    # Setup model and loss function
    model, loss_fn = setup_model(args, img_res, use_fp16, device, logger)

    # Log model information
    _ = ModelUtils.get_parameter_number(model, logger=logger)

    # Setup optimizer, scheduler, and training components
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    if device.type == "cuda":
        from torch.amp import GradScaler

        scaler = GradScaler("cuda")
        logger.info("GradScaler initialized for CUDA")
    else:
        scaler = None
        logger.info("GradScaler disabled (AMP not supported on CPU)")

    # Setup metrics tracking
    metric_names = ["MAE", "NMAE", "RMSE", "R2", "PEARSON", "KL"]
    # metric_funcs = {"MAE": mae_all, "NMAE": nmae_all, "RMSE": rmse_all, "R2": r2_all}

    # Initialize validation metrics with ALL expected keys from run_validation
    valid_metrics_keys = []
    for k in args.varnames_list:
        for m in metric_names:
            valid_metrics_keys.append(f"{k}_pred_vs_fine_{m}")  # Model predictions
            valid_metrics_keys.append(f"{k}_coarse_vs_fine_{m}")  # Coarse baselines
    for m in metric_names:
        valid_metrics_keys.append(f"average_pred_vs_fine_{m}")  # Overall averages
        valid_metrics_keys.append(
            f"average_coarse_vs_fine_{m}"
        )  # Overall coarse averages

    valid_metrics_history = {key: [] for key in valid_metrics_keys}
    train_loss_history = [0] * args.num_epochs
    valid_loss_history = [0] * args.num_epochs
    train_loss = MetricTracker()

    logger.info(f"Tracking metrics: {metric_names}")
    logger.info(f"Validation metrics: {list(valid_metrics_history.keys())}")

    # Setup training state tracking
    start_epoch = 0
    samples_processed = 0
    batches_processed = 0
    avg_val_loss = float("inf")
    best_val_loss = float("inf")
    avg_epoch_loss = float("inf")
    best_epoch = 0

    # Handle checkpoint loading if needed
    # if args.run_type in ["resume_train", "inference"]:
    if args.run_type in ["resume_train", "inference", "inference_regional"]:
        checkpoint_path = os.path.join(paths.checkpoints, args.load_checkpoint_name)
        if args.debug:
            logger.info("=" * 60)
            logger.info("CHECKPOINT LOADING DEBUG INFO")
            logger.info("=" * 60)
            logger.info(f"Run type: {args.run_type}")
            logger.info(f"Checkpoint path: {checkpoint_path}")
            logger.info(f"Model checkpoint directory: {paths.checkpoints}")
            logger.info(f"Load checkpoint name: {args.load_checkpoint_name}")
            logger.info(
                f"Full checkpoint path exists: {os.path.exists(checkpoint_path)}"
            )

        if os.path.exists(checkpoint_path):
            if args.debug:
                logger.info(f"Loading checkpoint from: {checkpoint_path}")
            (
                epoch,
                samples_processed,
                batches_processed,
                best_val_loss,
                best_epoch,
                checkpoint,
            ) = ModelUtils.load_training_checkpoint(
                checkpoint_path, model, optimizer, device, logger=logger
            )
            avg_val_loss = best_val_loss
            if args.debug:
                logger.info("Checkpoint loaded successfully")
                logger.info(f" └── Epoch: {epoch}")
                logger.info(f" └── Samples processed: {samples_processed:,}")
                logger.info(f" └── Batches processed: {batches_processed:,}")
                logger.info(f" └── Best validation loss: {best_val_loss:.6f}")
                logger.info(f" └── Best epoch: {best_epoch}")
                logger.info("Checkpoint keys available:")
                for key in checkpoint.keys():
                    if isinstance(checkpoint[key], (list, dict)):
                        if key in ["train_loss_history", "valid_loss_history"]:
                            logger.info(
                                f" └── {key}: list with {len(checkpoint[key])} elements"
                            )
                        elif key == "valid_metrics_history":
                            logger.info(
                                f" └── {key}: dict with {len(checkpoint[key])} keys"
                            )
                        elif key == "args":
                            logger.info(
                                f" └── {key}: dict with {len(checkpoint[key])} arguments"
                            )
                        else:
                            logger.info(f" └── {key}: {type(checkpoint[key]).__name__}")
                    else:
                        logger.info(f" └── {key}: {checkpoint[key]}")
            if args.run_type == "resume_train":
                start_epoch = epoch + 1
                if args.debug:
                    logger.info(f"Resuming training from epoch {start_epoch}")
                    logger.info(
                        f"Current train_loss_history length: {len(train_loss_history)}"
                    )
                    logger.info(
                        f"Current valid_loss_history length: {len(valid_loss_history)}"
                    )
                # Load history if available
                if "train_loss_history" in checkpoint:
                    train_loss_history[:start_epoch] = checkpoint["train_loss_history"][
                        :start_epoch
                    ]
                if "valid_loss_history" in checkpoint:
                    valid_loss_history[:start_epoch] = checkpoint["valid_loss_history"][
                        :start_epoch
                    ]
                if "valid_metrics_history" in checkpoint:
                    for key in valid_metrics_history:
                        if key in checkpoint["valid_metrics_history"]:
                            valid_metrics_history[key] = checkpoint[
                                "valid_metrics_history"
                            ][key]

                logger.info(f"Resuming training from epoch {start_epoch}")
            else:
                logger.info(f"Model loaded for {args.run_type}")

        else:
            logger.error(f"Checkpoint not found at: {checkpoint_path}")
            logger.error(
                f"Cannot do run type: {args.run_type} without checkpoint. Exiting."
            )
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Prepare for model saving
    if args.save_model:
        logger.info("Model saving enabled")
        # save_counter = 0

    # Setup GPU support
    if torch.cuda.is_available():
        model.cuda()
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)

    # ============================================================================
    # INFERENCE MODE - Run validation directly
    # ============================================================================
    # if args.run_type == "inference":
    if args.run_type in ["inference", "inference_regional"]:
        logger.info("=" * 60)
        logger.info("RUNNING INFERENCE/VALIDATION")
        logger.info("=" * 60)

        logger.info(
            f"Validation dataset temporal range: "
            f"{valid_dataset.stime} → {valid_dataset.etime} "
            f"(total timesteps = {valid_dataset.etime - valid_dataset.stime})"
        )

        if args.compute_crps and args.inference_type != "sampler":
            logger.warning(
                "CRPS requested but inference_type is not 'sampler'. "
                "CRPS requires probabilistic sampling. Disabling CRPS."
            )
            args.compute_crps = False

        # Run validation (which is essentially inference on validation data)
        assert (
            valid_loader is not None
        ), "Validation data loader must be available for inference"
        avg_val_loss, val_metrics = run_validation(
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
            epoch=0,
            writer=writer,
            plot_every_n_epochs=1,  # Always plot for inference
            paths=paths,
            compute_crps=args.compute_crps,  # True for diffusion models, False for unet
            crps_ensemble_size=args.crps_ensemble_size,
            crps_batch_size=args.crps_batch_size,
        )
        logger.info("Inference completed successfully!")
        exit(0)

    logger.info("Start training...")

    # Training loop with validation
    for epoch in range(start_epoch, args.num_epochs):
        train_dataset.new_epoch()
        model.train()

        train_loss.reset()
        previous_time = time.time()

        loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training Epoch {epoch}",
        )

        for batch_idx, batch in loop:
            # Move data to device
            features = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            # lat_batch = batch["corrdinates"]["lat"].to(device)
            # lon_batch = batch["corrdinates"]["lon"].to(device)
            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"batch idx:{batch_idx}, features shape:{features.shape}, targets shape:{targets.shape}"
                )

            # Prepare labels based on time normalization
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

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision training
            with torch.amp.autocast(device_type=device.type, dtype=torch_dtype):
                loss = loss_fn(model, targets, features, labels)
                loss = loss.mean()

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update loss trackers
            train_loss.update(loss.item(), targets.shape[0])

            # Calculate timing
            current_time = time.time()
            batch_time = current_time - previous_time
            previous_time = current_time

            # Update progress bar
            loop.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{train_loss.getmean():.4f}",
                    "Time": f"{batch_time:.2f}s",
                }
            )

        # End of epoch - Run validation if validation loader exists
        avg_epoch_loss = train_loss.getmean()
        train_loss_history[epoch] = avg_epoch_loss

        # TensorBoard logging for training
        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)

        # Run validation
        if valid_loader is not None:
            avg_val_loss, val_metrics = run_validation(
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
                writer,
                plot_every_n_epochs=10,
                paths=paths,
            )
            valid_loss_history[epoch] = avg_val_loss

            # Update validation metrics history
            for metric_name, tracker in val_metrics.items():
                if metric_name in valid_metrics_history:
                    valid_metrics_history[metric_name].append(tracker.getmean())
                else:
                    logger.warning(
                        f"Unexpected metric {metric_name} not found in valid_metrics_history"
                    )

        # Update scheduler based on mean of all validation MAE history
        if (
            valid_loader is not None
            and valid_metrics_history["average_pred_vs_fine_MAE"]
        ):
            # Calculate mean of all validation MAE values so far using Python's sum/len
            mae_history = valid_metrics_history["average_pred_vs_fine_MAE"]
            mean_val_mae = sum(mae_history) / len(mae_history)
            scheduler.step(mean_val_mae)
            logger.info(
                f"Scheduler step with mean validation MAE (all {len(mae_history)} epochs): {mean_val_mae:.4f}"
            )
        else:
            scheduler.step(avg_epoch_loss)

        # Log epoch results
        logger.info(f"Epoch {epoch} completed - Train Loss: {avg_epoch_loss:.4f}")
        if valid_loader is not None:
            logger.info(f"Epoch {epoch} completed - Val Loss: {avg_val_loss:.4f}")

        # Save epoch checkpoint (every 10 epochs)
        if args.save_model and epoch % 10 == 0:
            ModelUtils.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                samples_processed=samples_processed,
                batches_processed=batches_processed,
                train_loss_history=train_loss_history,
                valid_loss_history=valid_loss_history,
                valid_metrics_history=valid_metrics_history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                avg_val_loss=avg_val_loss if valid_loader is not None else 0.0,
                avg_epoch_loss=avg_epoch_loss,
                args=args,
                paths=paths,
                logger=logger,
                checkpoint_type="epoch",
                save_full_model=True,
            )
        # Save best model based on validation loss
        if valid_loader is not None and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

            ModelUtils.save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                samples_processed=samples_processed,
                batches_processed=batches_processed,
                train_loss_history=train_loss_history,
                valid_loss_history=valid_loss_history,
                valid_metrics_history=valid_metrics_history,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                avg_val_loss=avg_val_loss,
                avg_epoch_loss=avg_epoch_loss,
                args=args,
                paths=paths,
                logger=logger,
                checkpoint_type="best",
                save_full_model=True,
            )

    # Generate plots at the end of training
    logger.info("Generating training summary plots...")

    # Plot losses
    plot_loss_histories(
        train_loss_history,
        valid_loss_history,
        filename=f"training_validation_loss_{args.prefix}.png",
        save_dir=paths.results,
    )

    # Plot metrics (only validation metrics available)
    plot_metric_histories(
        valid_metrics_history,
        args.varnames_list,
        metric_names,
        filename=f"validation_metrics_{args.prefix}",
        save_dir=paths.results,
    )

    # Plot average metrics
    plot_average_metrics(
        valid_metrics_history,
        metric_names,
        filename=f"average_metrics_{args.prefix}.png",
        save_dir=paths.results,
    )

    logger.info("Training summary plots generated successfully!")
    logger.info("Generating spatiotemporal coverage plots...")

    # Plot training data
    plot_spatiotemporal_histograms(
        steps,
        tindex_lim=(train_dataset.stime, train_dataset.etime),
        centers=train_dataset.center_tracker,
        tindices=train_dataset.tindex_tracker,
        mode=train_dataset.mode,
        filename=f"{args.prefix}_",
        save_dir=paths.results,
    )

    # Plot validation data if available
    if valid_dataset is not None:
        plot_spatiotemporal_histograms(
            steps,
            tindex_lim=(valid_dataset.stime, valid_dataset.etime),
            centers=valid_dataset.center_tracker,
            tindices=valid_dataset.tindex_tracker,
            mode=valid_dataset.mode,
            filename=f"{args.prefix}_",
            save_dir=paths.results,
        )

    logger.info("Spatiotemporal coverage plots completed!")

    # Save final checkpoint at the end of training
    if args.save_model:
        ModelUtils.save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=args.num_epochs - 1,
            samples_processed=samples_processed,
            batches_processed=batches_processed,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            valid_metrics_history=valid_metrics_history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            avg_val_loss=avg_val_loss if valid_loader is not None else 0.0,
            avg_epoch_loss=avg_epoch_loss,
            args=args,
            paths=paths,
            logger=logger,
            checkpoint_type="final",
            save_full_model=True,
        )
        logger.info("Final model checkpoint saved successfully!")
    logger.info("Training process completed successfully!")


if __name__ == "__main__":
    main()
