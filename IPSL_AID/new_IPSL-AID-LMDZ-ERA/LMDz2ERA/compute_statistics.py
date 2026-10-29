import sys
# sys.path.insert(1,"C:/Users/pchapel/code/IPSL-AID/")
sys.path.insert(1,"/lustre/fswork/projects/rech/huw/uzb76ep/IPSL_AID/IPSL-AID/")
sys.path.insert(1,"/lustre/fswork/projects/rech/huw/uzb76ep/Downscaling_JZ/new_IPSL-AID-LMDZ-ERA/")
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
from main_lmdz_era import (
    parse_args,
    setup_directories_and_logging,
    log_configuration,
    setup_data_paths,
    setup_training_environment,
    create_data_loaders,
)
import json


def main():
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

    train_loader, img_res, train_dataset = create_data_loaders(
        args,
        paths,
        normalization_type,
        normalization_type_coarse,
        normalization_type_diagnostics,
        normalization_type_dynamic_cov,
        index_mapping,
        index_mapping_coarse,
        index_mapping_dynamic_cov,
        index_mapping_diagnostics,
        per_var_paths_train,
        per_coarse_var_paths_train,
        per_diagnostic_var_paths_train,
        per_dynamic_cov_paths_train,
        torch_dtype,
        np_dtype,
        logger,
        mode="train",
        run_type=args.run_type,
        multiplication_factor_precip_coarse=1000,
        multiplication_factor_precip_fine=1000,
    )
    logger.info(f"Training dataset loaded with image resolution: {img_res}")
    test_loader, img_res, test_dataset = create_data_loaders(
        args,
        paths,
        normalization_type,
        normalization_type_coarse,
        normalization_type_diagnostics,
        normalization_type_dynamic_cov,
        index_mapping,
        index_mapping_coarse,
        index_mapping_dynamic_cov,
        index_mapping_diagnostics,
        per_var_paths_valid,
        per_coarse_var_paths_valid,
        per_diagnostic_var_paths_valid,
        per_dynamic_cov_paths_valid,
        torch_dtype,
        np_dtype,
        logger,
        mode="validation",
        run_type=args.run_type,
        multiplication_factor_precip_coarse=1000,
        multiplication_factor_precip_fine=1000,
    )

    #iterate over all batches of train_loader to get coarse variables, fine variables and covariates.
    coarses_transform = []
    residuals = []
    covariates_transforms = []
    diagnostics_transforms = []
    logger.info("Train period : ")
    train_loop = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Seeing all batches in train_loader",
    )
    time_start = time.time()
    for batch_idx, batch in train_loop:
        coarse_transform = batch["coarse_transform"]
        coarses_transform.append(coarse_transform)
        residual = batch["residual"]
        residuals.append(residual)
        covariates_transform = batch["covariates_transform"]
        covariates_transforms.append(covariates_transform)
        diagnostics_transform = batch["diagnostics_transform"]
        diagnostics_transforms.append(diagnostics_transform)
    time_end = time.time()
    logger.info(f"Time to see the whole dataset : {time_end - time_start}")
    coarses_transform = torch.cat(coarses_transform, dim=0)
    logger.info(f"coarses_transform shape : {coarses_transform.shape}")
    means_coarses_transform = torch.mean(coarses_transform, dim=(0,2,3))
    stds_coarses_transform = torch.std(coarses_transform, dim=(0,2,3))
    logger.info(f"means coarses transform : {means_coarses_transform}")
    logger.info(f"stds coarses transform : {stds_coarses_transform}")
    normalisation_file_coarse = {}
    for i,var in enumerate(args.varnames_list) :
        key = ""
        if "log" in normalization_type[var] :
            key += "log_"
        key += var
        normalisation_file_coarse[key] = {}
        normalisation_file_coarse[key]["vmean"] = means_coarses_transform[i].item()
        normalisation_file_coarse[key]["vstd"] = stds_coarses_transform[i].item()
    logger.info("Normalisation file coarse :")
    logger.info(normalisation_file_coarse)
    logger.info(json.dumps(normalisation_file_coarse))

    residuals = torch.cat(residuals, dim=0)
    logger.info(f"residuals shape : {residuals.shape}")
    means_residuals = torch.mean(residuals, dim=(0,2,3))
    stds_residuals = torch.std(residuals, dim=(0,2,3))
    logger.info(f"means residuals : {means_residuals}")
    logger.info(f"stds residuals : {stds_residuals}")
    normalisation_file_residual = {}
    for i,var in enumerate(args.varnames_list) :
        key = ""
        if "log" in normalization_type[var] :
            key += "log_"
        key += var
        key += "_residual"
        normalisation_file_residual[key] = {}
        normalisation_file_residual[key]["vmean"] = means_residuals[i].item()
        normalisation_file_residual[key]["vstd"] = stds_residuals[i].item()
    logger.info("Normalisation file residuals :")
    logger.info(normalisation_file_residual)
    logger.info(json.dumps(normalisation_file_residual))

    covariates_transforms = torch.cat(covariates_transforms, dim=0)
    logger.info(f"covariates_transforms shape : {covariates_transforms.shape}")
    means_covariates_transforms = torch.mean(covariates_transforms, dim=(0,2,3))
    stds_covariates_transforms = torch.std(covariates_transforms, dim=(0,2,3))
    logger.info(f"means coarses transform : {means_covariates_transforms}")
    logger.info(f"stds std transform : {stds_covariates_transforms}")
    normalisation_file_coarse = {}
    for i,var in enumerate(args.dynamic_covariates) :
        key = ""
        if "log" in normalization_type_dynamic_cov[var] :
            key += "log_"
        key += var
        normalisation_file_coarse[key] = {}
        normalisation_file_coarse[key]["vmean"] = means_covariates_transforms[i].item()
        normalisation_file_coarse[key]["vstd"] = stds_covariates_transforms[i].item()
    logger.info("Normalisation file coarse :")
    logger.info(json.dumps(normalisation_file_coarse))

    diagnostics_transforms = torch.cat(diagnostics_transforms, dim=0)
    logger.info(f"diagnostics_transforms shape : {diagnostics_transforms.shape}")
    means_diagnostics_transforms = torch.mean(diagnostics_transforms, dim=(0,2,3))
    stds_diagnostics_transforms = torch.std(diagnostics_transforms, dim=(0,2,3))
    logger.info(f"means residuals : {means_diagnostics_transforms}")
    logger.info(f"stds residuals : {stds_diagnostics_transforms}")
    normalisation_file_residual = {}
    for i,var in enumerate(args.diagnostics_varnames_list) :
        key = ""
        if "log" in normalization_type_diagnostics[var] :
            key += "log_"
        key += var
        normalisation_file_residual[key] = {}
        normalisation_file_residual[key]["vmean"] = means_diagnostics_transforms[i].item()
        normalisation_file_residual[key]["vstd"] = stds_diagnostics_transforms[i].item()
    logger.info("Normalisation file residuals :")
    logger.info(json.dumps(normalisation_file_residual))

    logger.info("=================================")
    logger.info("=================================")

    #iterate over all batches of test_loader to get coarse variables, fine variables and covariates.
    coarses_transform = []
    residuals = []
    covariates_transforms = []
    diagnostics_transforms = []
    logger.info("Test period : ")
    test_loop = tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc=f"Seeing all batches in test_loader",
    )
    time_start = time.time()
    for batch_idx, batch in test_loop:
        coarse_transform = batch["coarse_transform"]
        coarses_transform.append(coarse_transform)
        residual = batch["residual"]
        residuals.append(residual)
        covariates_transform = batch["covariates_transform"]
        covariates_transforms.append(covariates_transform)
        diagnostics_transform = batch["diagnostics_transform"]
        diagnostics_transforms.append(diagnostics_transform)
    time_end = time.time()
    logger.info(f"Time to see the whole dataset : {time_end - time_start}")
    coarses_transform = torch.cat(coarses_transform, dim=0)
    logger.info(f"coarses_transform shape : {coarses_transform.shape}")
    means_coarses_transform = torch.mean(coarses_transform, dim=(0,2,3))
    stds_coarses_transform = torch.std(coarses_transform, dim=(0,2,3))
    logger.info(f"means coarses transform : {means_coarses_transform}")
    logger.info(f"stds coarses transform : {stds_coarses_transform}")
    normalisation_file_coarse = {}
    for i,var in enumerate(args.varnames_list) :
        key = ""
        if "log" in normalization_type[var] :
            key += "log_"
        key += var
        normalisation_file_coarse[key] = {}
        normalisation_file_coarse[key]["vmean"] = means_coarses_transform[i].item()
        normalisation_file_coarse[key]["vstd"] = stds_coarses_transform[i].item()
    logger.info("Normalisation file coarse :")
    logger.info(normalisation_file_coarse)
    logger.info(json.dumps(normalisation_file_coarse))

    residuals = torch.cat(residuals, dim=0)
    logger.info(f"residuals shape : {residuals.shape}")
    means_residuals = torch.mean(residuals, dim=(0,2,3))
    stds_residuals = torch.std(residuals, dim=(0,2,3))
    logger.info(f"means residuals : {means_residuals}")
    logger.info(f"stds residuals : {stds_residuals}")
    normalisation_file_residual = {}
    for i,var in enumerate(args.varnames_list) :
        key = ""
        if "log" in normalization_type[var] :
            key += "log_"
        key += var
        key += "_residual"
        normalisation_file_residual[key] = {}
        normalisation_file_residual[key]["vmean"] = means_residuals[i].item()
        normalisation_file_residual[key]["vstd"] = stds_residuals[i].item()
    logger.info("Normalisation file residuals :")
    logger.info(normalisation_file_residual)
    logger.info(json.dumps(normalisation_file_residual))

    covariates_transforms = torch.cat(covariates_transforms, dim=0)
    logger.info(f"covariates_transforms shape : {covariates_transforms.shape}")
    means_covariates_transforms = torch.mean(covariates_transforms, dim=(0,2,3))
    stds_covariates_transforms = torch.std(covariates_transforms, dim=(0,2,3))
    logger.info(f"means coarses transform : {means_covariates_transforms}")
    logger.info(f"stds std transform : {stds_covariates_transforms}")
    normalisation_file_coarse = {}
    for i,var in enumerate(args.dynamic_covariates) :
        key = ""
        if "log" in normalization_type_dynamic_cov[var] :
            key += "log_"
        key += var
        normalisation_file_coarse[key] = {}
        normalisation_file_coarse[key]["vmean"] = means_covariates_transforms[i].item()
        normalisation_file_coarse[key]["vstd"] = stds_covariates_transforms[i].item()
    logger.info("Normalisation file coarse :")
    logger.info(json.dumps(normalisation_file_coarse))

    diagnostics_transforms = torch.cat(diagnostics_transforms, dim=0)
    logger.info(f"diagnostics_transforms shape : {diagnostics_transforms.shape}")
    means_diagnostics_transforms = torch.mean(diagnostics_transforms, dim=(0,2,3))
    stds_diagnostics_transforms = torch.std(diagnostics_transforms, dim=(0,2,3))
    logger.info(f"means residuals : {means_diagnostics_transforms}")
    logger.info(f"stds residuals : {stds_diagnostics_transforms}")
    normalisation_file_residual = {}
    for i,var in enumerate(args.diagnostics_varnames_list) :
        key = ""
        if "log" in normalization_type_diagnostics[var] :
            key += "log_"
        key += var
        normalisation_file_residual[key] = {}
        normalisation_file_residual[key]["vmean"] = means_diagnostics_transforms[i].item()
        normalisation_file_residual[key]["vstd"] = stds_diagnostics_transforms[i].item()
    logger.info("Normalisation file residuals :")
    logger.info(json.dumps(normalisation_file_residual))

    logger.info("=================================")
    logger.info("=================================")


if __name__ == "__main__":
    main()
