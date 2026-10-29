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

import sys
# sys.path.insert(1,"C:/Users/pchapel/code/IPSL-AID/")
sys.path.insert(1,"/lustre/fswork/projects/rech/huw/uzb76ep/IPSL_AID/IPSL-AID/")
import os
import time
import argparse
from IPSL_AID.logger import Logger
import IPSL_AID.networks
from IPSL_AID.utils import FileUtils, EasyDict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from IPSL_AID.dataset import stats
from dataset import DataPreprocessorLMDZ2ERA
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from model import load_model_and_loss
from IPSL_AID.model_utils import ModelUtils
import torch.optim as optim
import xarray as xr
import evaluater

from IPSL_AID.diagnostics import (
    plot_metric_histories,
    plot_loss_histories,
    plot_average_metrics,
    plot_spatiotemporal_histograms,
)

from IPSL_AID.evaluater import (
    MetricTracker
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
        "--domain_lon",
        type=str,
        nargs="+",
        default=["-6","9.75"]
    )
    parser.add_argument(
        "--domain_lat",
        type=str,
        nargs="+",
        default=["54.75","39"]
    )
    # Run configuration
    parser.add_argument(
        "--run_type",
        type=str,
        default="train",
        choices=["train", "resume_train", "inference"],
        help="Run type: 'train', 'resume_train', 'inference'",
    )
    parser.add_argument(
        "--num_members",
        type=int,
        default=1,
        help="number of ensemble members to generate. Used only in inference mode.",
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
        "--coarse_varnames_list",
        type=str,
        nargs="+",
        default=["VAR_2T", "VAR_10U", "VAR_10V"],
        help="List of variable names to train on",
    )
    parser.add_argument(
        "--diagnostics_varnames_list",
        type=str,
        nargs="+",
        default=None,
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
        "--normalization_types_coarse",
        type=str,
        nargs="+",
        default=["VAR_2T=standard", "VAR_10U=standard", "VAR_10V=standard"],
        help="Normalization types for each variable as 'var=type' pairs",
    )
    parser.add_argument(
        "--normalization_types_dynamic_cov",
        type=str,
        nargs="+",
        default=None,
        help="Normalization types for each variable as 'var=type' pairs",
    )
    parser.add_argument(
        "--normalization_types_diagnostics",
        type=str,
        nargs="+",
        default=None,
        help="Normalization types for each variable as 'var=type' pairs",
    )
    parser.add_argument(
        "--path_folder_stats",
        type=str,
        default='./stats/',
        help="Directory containing json files with normalisation constants"
    )
    parser.add_argument(
        "--json_stats_rawdata",
        type=str,
        default="statistics_lmdz_nudg_6h_1_1985-2014.json",
        help="Path to json file containing normalisation constants for coarse variables and dynamic covariates",
    )
    parser.add_argument(
        "--json_stats_residual",
        type=str,
        default="statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        help="Path to json file containing normalisation constants for residuals",
    )
    parser.add_argument(
        "--json_stats_diag",
        type=str,
        default="statistics_era5_lmdz_nudg_6h_1_1985-2014.json",
        help="Path to json file containing normalisation constants for diags",
    )
    parser.add_argument(
        "--dynamic_covariates",
        nargs="+",
        type=str,
        default=None,
        help="List of dynamic covariates",
    )
    parser.add_argument(
        "--datadir_dynamic_cov",
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

    # Training configuration
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--datadir", type=str, required=True, help="Dataset path")
    parser.add_argument("--datadir_diagnostics", type=str, default=None, help="Dataset path")
    parser.add_argument(
        "--per_var_datadir_train",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_var_datadir_valid",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument("--datadir_coarse", type=str, default=None, help="Coarse Dataset path if required")
    parser.add_argument(
        "--per_coarse_var_datadir_train",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_coarse_var_datadir_valid",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_dynamic_cov_datadir_train",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_dynamic_cov_datadir_valid",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_diagnostic_var_datadir_train",
        type=str,
        nargs="+",
        default=None,
        help="Per-variable data directories as VAR=path pairs",
    )
    parser.add_argument(
        "--per_diagnostic_var_datadir_valid",
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
        "--multiplication_factor_precip_coarse", type=float, default=1000.0, help="multiplication factor for coarse precip. should be 1000 for ERA5, 3600 for LMDz"
    )
    parser.add_argument(
        "--multiplication_factor_precip_fine", type=float, default=1000.0, help="multiplication factor for fine precip. should be 1000 for ERA5."
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Beta parameter for loss function"
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
        "--custom_loss",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use custom loss instead of vanilla MSE",
    )
    parser.add_argument(
        "--weight_facile",
        type=float,
        default=1,
        help="weight for loss on easy variables",
    )
    parser.add_argument(
        "--weight_cape",
        type=float,
        default=1,
        help="weight for loss on cape variable",
    )
    parser.add_argument(
        "--weight_precip",
        type=float,
        default=1,
        help="weight for loss on precip variable",
    )
    parser.add_argument(
        "--weight_precip_wasserstein",
        type=float,
        default=0,
        help="weight for wasserstein term in loss on precip",
    )
    parser.add_argument(
        "--weight_occurence_intensity_loss",
        type=float,
        default=0,
        help="weight for occurence / intensity in loss on precip",
    )
    parser.add_argument(
        "--quantiles_wasserstein",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        help="quantiles to regularize in loss",
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
    # CRPS args
    parser.add_argument(
        "--compute_crps", type=lambda x: x.lower() == "true", default=False
    )

    parser.add_argument("--crps_ensemble_size", type=int, default=10)
    parser.add_argument("--crps_batch_size", type=int, default=2)

    return parser.parse_args()


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

        main_folder/sub_folder/logs
        main_folder/sub_folder/results
        main_folder/sub_folder/runs
        main_folder/sub_folder/checkpoints
        main_folder/sub_folder/stats

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
    paths.logs = os.path.join(args.main_folder, args.sub_folder, "logs")
    paths.results = os.path.join(
        args.main_folder, args.sub_folder, "results"
    )
    paths.runs = os.path.join(args.main_folder, args.sub_folder, "runs")
    paths.checkpoints = os.path.join(
        args.main_folder, args.sub_folder, "checkpoints"
    )
    paths.stats = os.path.join(args.main_folder, args.sub_folder, "stats")
    paths.stats_dir = os.path.join(args.main_folder, args.sub_folder, "stats")
    paths.datadir = args.datadir
    paths.datadir_coarse = args.datadir_coarse
    paths.datadir_dynamic_cov = args.datadir_dynamic_cov
    paths.datadir_diagnostics = args.datadir_diagnostics
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
    logger.info(f" └── Dynamic covariates dir: '{args.datadir_dynamic_cov}'")
    logger.info(
        f" └── Diagnostic variables: {args.diagnostics_varnames_list if args.diagnostics_varnames_list else 'None'}"
    )
    logger.info(f" └── Diagnostic variables dir: '{args.datadir_diagnostics}'")
    logger.info(f" └── Units list: {args.units_list}")
    logger.info(f" └── Normalization types: {args.normalization_types}")
    logger.info(f" └── Data directory: '{args.datadir}'")
    logger.info(f" └── Coarse Data directory: '{args.datadir_coarse if args.datadir_coarse else args.datadir}'")

    # Time range configuration
    logger.info("\nTime Range Configuration:")
    logger.info(f" └── Time normalization: '{args.time_normalization}'")

    # Training configuration
    logger.info("\nTraining Configuration:")
    logger.info(f" └── Number of epochs: {args.num_epochs}")
    logger.info(f" └── Batch size: {args.batch_size}")
    logger.info(f" └── Number of workers: {args.num_workers}")
    logger.info(f" └── Learning rate: {args.learning_rate}")
    logger.info(f" └── Dtype: '{args.dtype}'")

    # Data processing parameters
    logger.info("\nData Processing Parameters:")
    logger.info(f" └── Beta: {args.beta}")

    # Model architecture
    logger.info("\nModel Architecture:")
    logger.info(f" └── Architecture: '{args.arch}'")
    logger.info(f" └── Preconditioner: '{args.precond}'")
    logger.info(f" └── Input channels: {args.in_channels}")
    logger.info(f" └── Conditioning channels: {args.cond_channels}")
    logger.info(f" └── Output channels: {args.out_channels}")

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
    if args.per_var_datadir_train is not None :
        logger.info("Fine train :")
        for item in args.per_var_datadir_train:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_var_datadir_valid is not None :
        logger.info("Fine valid :")
        for item in args.per_var_datadir_valid:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_coarse_var_datadir_train is not None :
        logger.info("Coarse train :")
        for item in args.per_coarse_var_datadir_train:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_coarse_var_datadir_valid is not None :
        logger.info("Coarse valid :")
        for item in args.per_coarse_var_datadir_valid:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_diagnostic_var_datadir_train is not None :
        logger.info("Diagnostics train :")
        for item in args.per_diagnostic_var_datadir_train:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_diagnostic_var_datadir_valid is not None :
        logger.info("Diagnostics valid :")
        for item in args.per_diagnostic_var_datadir_valid:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_dynamic_cov_datadir_train is not None :
        logger.info("Dynamic covariates train :")
        for item in args.per_dynamic_cov_datadir_train:
            var, path = item.split("=")
            logger.info(f" └── {var}: '{path}'")
    if args.per_dynamic_cov_datadir_valid is not None :
        logger.info("Dynamic covariates valid :")
        for item in args.per_dynamic_cov_datadir_valid:
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
    elif args.run_type == "inference_regional":
        logger.info(" └── Mode: Regional inference")

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
    # ------------------------------------------------------------------
    # Per-variable data paths configuration (using EasyDict)
    # ------------------------------------------------------------------
    # Default path is used as a fallback when a variable-specific path
    # is not provided via the command line.
    per_var_paths_train = EasyDict()
    per_var_paths_train.default = paths.datadir
    per_var_paths_valid = EasyDict()
    per_var_paths_valid.default = paths.datadir
    per_coarse_var_paths_train = EasyDict()
    per_coarse_var_paths_train.default = paths.datadir_coarse
    per_coarse_var_paths_valid = EasyDict()
    per_coarse_var_paths_valid.default = paths.datadir_coarse
    per_diagnostic_var_paths_train = EasyDict()
    per_diagnostic_var_paths_train.default = paths.datadir_diagnostics
    per_diagnostic_var_paths_valid = EasyDict()
    per_diagnostic_var_paths_valid.default = paths.datadir_diagnostics
    per_dynamic_cov_paths_train = EasyDict()
    per_dynamic_cov_paths_train.default = paths.datadir_dynamic_cov
    per_dynamic_cov_paths_valid = EasyDict()
    per_dynamic_cov_paths_valid.default = paths.datadir_dynamic_cov

    # Per-variable data directories passed as VAR=path
    if args.per_var_datadir_train is not None:
        for item in args.per_var_datadir_train:
            var, path = item.split("=")
            per_var_paths_train[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
    if args.per_var_datadir_valid is not None:
        for item in args.per_var_datadir_valid:
            var, path = item.split("=")
            per_var_paths_valid[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")

    logger.info(f"[Data paths train] default → {per_var_paths_train.default}")
    logger.info(f"[Data paths valid] default → {per_var_paths_valid.default}")

    if args.per_coarse_var_datadir_train is not None:
        for item in args.per_coarse_var_datadir_train:
            var, path = item.split("=")
            per_coarse_var_paths_train[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
        logger.info(f"[Coarse Data paths train] default → {per_coarse_var_paths_train.default}")
    if args.per_coarse_var_datadir_valid is not None:
        for item in args.per_coarse_var_datadir_valid:
            var, path = item.split("=")
            per_coarse_var_paths_valid[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
        logger.info(f"[Coarse Data paths valid] default → {per_coarse_var_paths_valid.default}")

    # diagnostic variables :
    if args.per_diagnostic_var_datadir_train is not None:
        for item in args.per_diagnostic_var_datadir_train:
            var, path = item.split("=")
            per_diagnostic_var_paths_train[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
        logger.info(f"[Diagnostic Data paths train] default → {per_diagnostic_var_paths_train.default}")
    if args.per_diagnostic_var_datadir_valid is not None:
        for item in args.per_diagnostic_var_datadir_valid:
            var, path = item.split("=")
            per_diagnostic_var_paths_valid[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
        logger.info(f"[Diagnostic Data paths valid] default → {per_diagnostic_var_paths_valid.default}")

    # Per-variable data directories passed as VAR=path
    if args.per_dynamic_cov_datadir_train is not None:
        for item in args.per_dynamic_cov_datadir_train:
            var, path = item.split("=")
            per_dynamic_cov_paths_train[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
    logger.info(f"[Dynamic covariates Data paths train] default → {per_dynamic_cov_paths_train.default}")
    # Per-variable data directories passed as VAR=path
    if args.per_dynamic_cov_datadir_valid is not None:
        for item in args.per_dynamic_cov_datadir_valid:
            var, path = item.split("=")
            per_dynamic_cov_paths_valid[var] = path
            # logger.info(f"Per-variable path: {var} → {path}")
    logger.info(f"[Dynamic covariates Data paths train] default → {per_dynamic_cov_paths_valid.default}")
    
    normalization_type = EasyDict()
    for mapping in args.normalization_types:
        if "=" in mapping:
            var_name, norm_type = mapping.split("=")
            normalization_type[var_name] = norm_type
        else:
            logger.warning(
                f"Invalid normalization mapping: {mapping}. Expected 'VAR_NAME=type'"
            )
    normalization_type_coarse = EasyDict()
    for mapping in args.normalization_types_coarse:
        if "=" in mapping:
            var_name, norm_type = mapping.split("=")
            normalization_type_coarse[var_name] = norm_type
        else:
            logger.warning(
                f"Invalid normalization mapping: {mapping}. Expected 'VAR_NAME=type'"
            )
    normalization_type_diagnostics = EasyDict()
    if args.normalization_types_diagnostics is not None :
        for mapping in args.normalization_types_diagnostics:
            if "=" in mapping:
                var_name, norm_type = mapping.split("=")
                normalization_type_diagnostics[var_name] = norm_type
            else:
                logger.warning(
                    f"Invalid normalization mapping: {mapping}. Expected 'VAR_NAME=type'"
                )
    normalization_type_dynamic_cov = EasyDict()
    if args.normalization_types_dynamic_cov is not None :
        for mapping in args.normalization_types_dynamic_cov:
            if "=" in mapping:
                var_name, norm_type = mapping.split("=")
                normalization_type_dynamic_cov[var_name] = norm_type
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

    for var in args.coarse_varnames_list:
        if var not in normalization_type_coarse:
            logger.warning(
                f"Variable '{var}' not found in normalization_types_coarse. Defaulting to 'standard'"
            )
            normalization_type_coarse[var] = "standard"

    if args.diagnostics_varnames_list is not None :
        for var in args.diagnostics_varnames_list:
            if var not in normalization_type_diagnostics:
                logger.warning(
                    f"Variable '{var}' not found in normalization_types_diagnostics. Defaulting to 'standard'"
                )
                normalization_type_diagnostics[var] = "standard"

    if args.dynamic_covariates is not None :
        for var in args.dynamic_covariates:
            if var not in normalization_type_dynamic_cov:
                logger.warning(
                    f"Variable '{var}' not found in normalization_types_dynamic_cov. Defaulting to 'standard'"
                )
                normalization_type_dynamic_cov[var] = "standard"

    logger.info(f"Normalization types: {normalization_type}")

    # Create index mapping
    index_mapping = {var_name: i for i, var_name in enumerate(args.varnames_list)}
    for var_name, idx in index_mapping.items():
        logger.info(f"{var_name}: Index {idx}")

    index_mapping_coarse = {var_name: i for i, var_name in enumerate(args.coarse_varnames_list)}
    for var_name, idx in index_mapping_coarse.items():
        logger.info(f"{var_name}: Index {idx}")

    if args.diagnostics_varnames_list is not None :
        index_mapping_diagnostics = {var_name: i for i, var_name in enumerate(args.diagnostics_varnames_list)}
    else :
        index_mapping_diagnostics = {}
    for var_name, idx in index_mapping_diagnostics.items():
        logger.info(f"{var_name}: Index {idx}")
    
    if args.dynamic_covariates is not None :
        index_mapping_dynamic_cov = {var_name: i for i, var_name in enumerate(args.dynamic_covariates)}
    else :
        index_mapping_dynamic_cov = {}
    for var_name, idx in index_mapping_dynamic_cov.items():
        logger.info(f"{var_name}: Index {idx}")

    return (
        normalization_type, normalization_type_coarse, normalization_type_dynamic_cov, normalization_type_diagnostics,
        index_mapping, index_mapping_coarse, index_mapping_dynamic_cov,index_mapping_diagnostics,
        per_var_paths_train, per_var_paths_valid,
        per_coarse_var_paths_train, per_coarse_var_paths_valid,
        per_dynamic_cov_paths_train, per_dynamic_cov_paths_valid,
        per_diagnostic_var_paths_train, per_diagnostic_var_paths_valid
    )



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
    # TODO : change seed on the fly when generating ensembles.
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
    normalization_type,
    normalization_type_coarse,
    normalization_type_diagnostics,
    normalization_type_dynamic_cov,
    index_mapping,
    index_mapping_coarse, # TODO : remove useless arguments in func signature
    index_mapping_diagnostics,
    index_mapping_dynamic_cov,
    per_var_paths,
    per_coarse_var_paths,
    per_diagnostics_var_paths,
    per_dynamic_cov_paths,
    torch_dtype,
    np_dtype,
    logger,
    mode="train",
    run_type="train",
    multiplication_factor_precip_coarse = 1,
    multiplication_factor_precip_fine = 1000,
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

    logger.info(f"Creating {mode} dataset:")

    # Create dataset with pre-loaded data
    lon_slice_domain = slice(
        float(args.domain_lon[0]),
        float(args.domain_lon[1])
    )
    lat_slice_domain = slice(
        float(args.domain_lat[0]),
        float(args.domain_lat[1])
    )
    logger.info(f"domain lon : {lon_slice_domain}")   
    logger.info(f"domain lat : {lat_slice_domain}")


    dataset = DataPreprocessorLMDZ2ERA(
        data_dir_coarse = paths.datadir_coarse,
        data_dir_fine = paths.datadir,
        data_dir_diagnostics = paths.datadir_diagnostics,
        files_coarse = (
            [ per_coarse_var_paths[var]  for var in args.coarse_varnames_list ] if args.coarse_varnames_list is not None else []
        ),
        files_fine = (
            [ per_var_paths[var] for var in args.varnames_list ] if args.varnames_list is not None else []
        ),
        files_diagnostics = (
            [ per_diagnostics_var_paths[var] for var in args.diagnostics_varnames_list ] if args.diagnostics_varnames_list is not None else []
        ),
        logger=logger,
        variables_name_coarse = args.coarse_varnames_list if args.coarse_varnames_list is not None else [],
        variables_name_fine = args.varnames_list if args.varnames_list is not None else [],
        variables_name_diagnostics = args.diagnostics_varnames_list if args.diagnostics_varnames_list is not None else [],
        path_folder_stats=args.path_folder_stats,
        json_stats_rawdata=args.json_stats_rawdata,
        json_stats_residual=args.json_stats_residual,
        json_stats_diag=args.json_stats_diag,
        dynamic_covariates_files=(
            [ per_dynamic_cov_paths[var] for var in args.dynamic_covariates ] if args.dynamic_covariates is not None else []
        ),
        dynamic_covariates_name = args.dynamic_covariates if args.dynamic_covariates is not None else [],
        dynamic_covariates_dir = paths.datadir_dynamic_cov,
        constant_variables_name=args.constant_varnames_list if args.constant_varnames_list is not None else [],
        files_constant_variables=(
            [args.constant_varnames_file for var in args.constant_varnames_list] if args.constant_varnames_list is not None else []
        ),
        constant_variables_dir = paths.datadir,
        lon_slice_domain = lon_slice_domain,
        lat_slice_domain = lat_slice_domain,
        normalization=args.time_normalization,
        normalization_type = normalization_type,
        normalization_type_coarse = normalization_type_coarse,
        normalization_type_diagnostics = normalization_type_diagnostics,
        normalization_type_dynamic_cov = normalization_type_dynamic_cov,
        multiplication_factor_precip_coarse = multiplication_factor_precip_coarse,
        multiplication_factor_precip_fine = multiplication_factor_precip_fine,
    )

    if mode == "train" :
        shuffle = True
    else :
        shuffle = False
    # Create data loader - set num_workers=0 since data is pre-loaded
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle= shuffle,
        num_workers=args.num_workers,  # Data is pre-loaded, no workers needed
        pin_memory=True,
        prefetch_factor = 2,
        drop_last=False # do last batch even if incomplete (needed for corrdiff approach)
    )
    img_res = dataset.input_shape

    logger.info(f"  {mode} dataset size: {len(dataset)}")
    logger.info(f"  {mode} data loader batches: {len(data_loader)}")

    return data_loader, img_res, dataset


def setup_model(args, img_res, use_fp16, device, logger, target_quantile_values = None):
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

    model, loss_fn = load_model_and_loss(
        opts,
        custom_loss= args.custom_loss,
        logger=logger, 
        device=device,
        weight_facile=args.weight_facile,
        weight_cape = args.weight_cape,
        weight_precip = args.weight_precip,
        weight_precip_wasserstein = args.weight_precip_wasserstein,
        weight_occurence_intensity_loss = args.weight_occurence_intensity_loss,
        target_quantile_values = target_quantile_values,
        quantiles = args.quantiles_wasserstein,
        beta = 50
    )

    return model, loss_fn

## for Wasserstein loss :
def compute_ref_quantiles(dataloader, channel = 5, quantiles = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], n_samples = 1e7):
    """
    Échantillonne n_samples valeurs de précipitations > 0
    depuis le jeu d'entraînement pour construire la distribution de référence.
    """
    samples = []
    for batch in dataloader:
        precip = batch['targets'][:,channel,:,:].flatten()
        samples.append(precip)
        if sum(len(s) for s in samples) >= n_samples:
            break

    ref = torch.cat(samples)[:int(n_samples)]
    quantile_values_ref = []
    for quantile in quantiles :
        quantile_value = torch.quantile(ref.flatten(), quantile)
        quantile_values_ref.append(quantile_value)
    print(
        f"Référence : {len(ref)} samples, " +
        f"max={ref.max():.1f}, " +
        (", ").join([ f"p{quantile}={quantile_value:.1f}" for quantile, quantile_value in zip(quantiles, quantile_values_ref)])
    )
    return quantile_values_ref

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
    # Parse command line arguments
    args = parse_args()

    # Setup directories and logging
    paths, logger = setup_directories_and_logging(args)

    # Log configuration parameters
    log_configuration(args, paths, logger)

    # Setup data paths and normalization statistics
    (
        normalization_type,
        normalization_type_coarse,
        normalization_type_dynamic_cov,
        normalization_type_diagnostics,
        index_mapping,
        index_mapping_coarse,
        index_mapping_dynamic_cov,
        index_mapping_diagnostics,
        per_var_paths_train,
        per_var_paths_valid,
        per_coarse_var_paths_train,
        per_coarse_var_paths_valid,
        per_dynamic_cov_paths_train,
        per_dynamic_cov_paths_valid,
        per_diagnostic_var_paths_train,
        per_diagnostic_var_paths_valid
    ) = setup_data_paths(args, paths, logger)

    # Setup training environment (device, data types, random seeds)
    device, torch_dtype, np_dtype, use_fp16 = setup_training_environment(args, logger)

    # Setup TensorBoard for visualization
    # if args.run_type != "inference":
    if args.run_type not in ["inference", "inference_regional"]:
        writer = SummaryWriter(f"{args.main_folder}/{args.sub_folder}/runs/")
        logger.info(
            f"TensorBoard enabled at: {args.main_folder}/{args.sub_folder}/runs/"
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
            normalization_type,
            normalization_type_coarse,
            normalization_type_diagnostics,
            normalization_type_dynamic_cov,
            index_mapping,
            index_mapping_coarse,
            index_mapping_diagnostics,
            index_mapping_dynamic_cov,
            per_var_paths_train,
            per_coarse_var_paths_train,
            per_diagnostic_var_paths_train,
            per_dynamic_cov_paths_train,
            torch_dtype,
            np_dtype,
            logger,
            mode="train",
            run_type=args.run_type,
            multiplication_factor_precip_coarse = args.multiplication_factor_precip_coarse,
            multiplication_factor_precip_fine = args.multiplication_factor_precip_fine,
        )
        logger.info(f"Training dataset loaded with image resolution: {img_res}")
        if args.custom_loss :
            quantiles_values_ref = compute_ref_quantiles(train_loader, channel = 5, quantiles = args.quantiles_wasserstein, n_samples = 1e7)
    else:
        logger.info("Inference mode: Skipping training data loader creation")
        train_loader, img_res, train_dataset = None, None, None

    valid_loader, valid_img_res, valid_dataset = create_data_loaders(
        args,
        paths,
        normalization_type,
        normalization_type_coarse,
        normalization_type_diagnostics,
        normalization_type_dynamic_cov,
        index_mapping,
        index_mapping_coarse,
        index_mapping_diagnostics,
        index_mapping_dynamic_cov,
        per_var_paths_valid,
        per_coarse_var_paths_valid,
        per_diagnostic_var_paths_valid,
        per_dynamic_cov_paths_valid,
        torch_dtype,
        np_dtype,
        logger,
        mode="validation",
        run_type=args.run_type
    )
    logger.info(f"Validation dataset loaded with image resolution: {valid_img_res}")
    if args.custom_loss :
        quantiles_values_ref = compute_ref_quantiles(valid_loader, channel = 5, quantiles = args.quantiles_wasserstein, n_samples = 1e7)
    # if args.run_type == "inference":
    if args.run_type in ["inference"]:
        img_res = valid_img_res  # Use validation image resolution for inference

    if args.run_type == "inference" and valid_loader is None:
        logger.error(
            "Inference mode requires a validation dataset, but none was created."
        )
        raise RuntimeError("Cannot run inference without a validation dataset.")

    # Setup model and loss function
    if args.custom_loss :
        model, loss_fn = setup_model(args, img_res, use_fp16, device, logger, target_quantile_values=quantiles_values_ref)
    else :
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
    if args.diagnostics_varnames_list is not None :
        for k in args.diagnostics_varnames_list:
            for m in metric_names:
                valid_metrics_keys.append(f"{k}_pred_vs_fine_{m}")  # Model predictions
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

        # Run validation (which is essentially inference on validation data)
        assert (
            valid_loader is not None
        ), "Validation data loader must be available for inference"
        for member in range(int(args.num_members)) :
            #TODO : modify random seed for different inferences to have different outputs (if necessary)
            save_reference = (member == 0)
            avg_val_loss, val_metrics = evaluater.run_validation(
                model,
                valid_dataset,
                valid_loader,
                loss_fn,
                index_mapping,
                index_mapping_diagnostics,
                args,
                device,
                logger,
                epoch=0,
                writer=writer,
                plot_every_n_epochs=1,  # Always plot for inference
                paths=paths,
                compute_crps=args.compute_crps,  # True for diffusion models, False for unet
                crps_batch_size=args.crps_batch_size,
                crps_ensemble_size=args.crps_ensemble_size,
                save_predicted = True,
                save_reference = save_reference,
                filename_suffix = f"test_save_validation_member_{member}"
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
        print(len(train_loader))
        loop = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Training Epoch {epoch}",
        )

        for batch_idx, batch in loop:
            # Move data to device
            features = batch["inputs"].to(device, non_blocking = True)
            targets = batch["targets"].to(device, non_blocking = True)
            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"batch idx:{batch_idx}, features shape:{features.shape}, targets shape:{targets.shape}"
                )

            # Prepare labels based on time normalization
            if args.time_normalization == "linear":
                labels = torch.stack(
                    (batch["doy"].to(device, non_blocking = True), batch["hour"].to(device, non_blocking = True)), dim=1
                )
            elif args.time_normalization == "cos_sin":
                labels = torch.stack(
                    (
                        batch["doy_sin"].to(device, non_blocking = True),
                        batch["doy_cos"].to(device, non_blocking = True),
                        batch["hour_sin"].to(device, non_blocking = True),
                        batch["hour_cos"].to(device, non_blocking = True),
                    ),
                    dim=1,
                )

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision training
            with torch.amp.autocast(device_type=device.type, dtype=torch_dtype):
                loss = loss_fn(model, targets, features, labels)
                #check if loss contains NaN :
                if torch.any(torch.isnan(loss)) :
                    logger.warning(f"loss contains nan for samples {batch['time_index']} !")
                #check if loss contains inf :
                if torch.any(torch.isinf(loss)) :
                    logger.warning(f"loss contains inf for batch_index {batch_idx}, with samples {batch['time_index']} !")
                loss = loss.mean()
                
            # Backward pass with gradient scaling
            if scaler is not None :
                scaler.scale(loss).backward()
                # clip gradients to stabilize training :
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else :
                loss.backward()

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
            avg_val_loss, val_metrics = evaluater.run_validation(
                model,
                valid_dataset,
                valid_loader,
                loss_fn,
                index_mapping,
                index_mapping_diagnostics,
                args,
                device,
                logger,
                epoch,
                writer,
                plot_every_n_epochs=1,
                paths=paths,
                compute_crps=args.compute_crps,  # True for diffusion models, False for unet
                crps_batch_size=args.crps_batch_size,
                crps_ensemble_size=args.crps_ensemble_size,
                save_predicted = False,
                filename_suffix = f"test_save_train_{epoch}"
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
        # save model if last epoch :
        if args.save_model and epoch == args.num_epochs - 1 :
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

    # Generate plots at the end of training
    logger.info("Generating training summary plots...")




if __name__ == "__main__":
    main()