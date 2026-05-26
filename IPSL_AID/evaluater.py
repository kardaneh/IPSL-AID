import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr

from IPSL_AID.diagnostics import (
    plot_validation_hexbin,
    plot_comparison_hexbin,
    plot_validation_pdfs,
    plot_power_spectra,
    plot_qq_quantiles,
    plot_surface,
    plot_MAE_map,
    plot_metrics_heatmap,
    plot_validation_mvcorr,
    plot_validation_mvcorr_space,
    plot_temporal_series_comparison,
)
from IPSL_AID.evaluater import (
    MetricTracker, mae_all, nmae_all,
    crps_ensemble_all, rmse_all, r2_all,
    pearson_all, kl_divergence_all, edm_sampler,
    sampler, generate_residuals_norm
    )

def reconstruct_original_layout(
    epoch, args, paths, all_data, dataset, device, logger
) :
    pass

def save_generated_data(
    all_data,
    vars,
    logger,
    filename_suffix,
    save_dir,
    save_reference = True
) :
    logger.info("saving generated data :")
    predictions = np.concat(all_data["predictions"], axis=0)
    diagnostics = np.concat(all_data["diagnostic"], axis=0)
    coarses = np.concat(all_data["coarse"], axis=0)
    fines = np.concat(all_data["fine"], axis=0)
    longitudes_fine = all_data["longitude_fine"][0][0] # access first tensor of shape [B,H] in list of ndarrays, then select first row of tensor to have tensor of shape [H,]
    latitudes_fine = all_data["latitude_fine"][0][0] # idem
    times = np.concat(all_data["times"], axis=0)
    # log dimensions :
    logger.info(f"predictions saved: shape {predictions.shape}")
    logger.info(f"coarses saved: shape {coarses.shape}")
    logger.info(f"fines saved: shape {fines.shape}")
    logger.info(f"longitudes_fine saved: shape {longitudes_fine.shape}")
    logger.info(f"latitudes_fine saved: shape {latitudes_fine.shape}")
    logger.info(f"times saved: shape {times.shape}")
    #reconstruct a xarray dataset for prediction :
    ds = xr.Dataset(
        data_vars = {
            var: (
                ["time","longitude","latitude"], #dimensions
                predictions[:,i,:,:], #data
                None #attributes
            ) for i,var in enumerate(vars)
        },
        coords = {
            "time": times,
            "longitude": longitudes_fine,
            "latitude": latitudes_fine,
        }
    )
    # save :
    np.save(
        os.path.join(save_dir, "predictions_" + filename_suffix + ".npy"),
        predictions
    )
    if save_reference :
        np.save(
            os.path.join(save_dir, "coarses_" + filename_suffix + ".npy"),
            coarses
        )
        np.save(
            os.path.join(save_dir, "fines_" + filename_suffix + ".npy"),
            fines
        )
        np.save(
            os.path.join(save_dir, "diagnostics_" + filename_suffix + ".npy"),
            diagnostics
        )
        np.save(
            os.path.join(save_dir, "longitudes_fine_" + filename_suffix + ".npy"),
            longitudes_fine
        )
        np.save(
            os.path.join(save_dir, "latitudes_fine_" + filename_suffix + ".npy"),
            latitudes_fine
        )
        np.save(
            os.path.join(save_dir, "times_" + filename_suffix + ".npy"),
            times
        )
    ds.to_netcdf(
        os.path.join(save_dir, "predictions_" + filename_suffix + ".nc")
    )

def generate(
    model,
    dataset,
    features,
    coarses,
    labels,
    targets,
    loss_fn,
    args,
    device,
    logger,
    epoch=0,
    batch_idx=0,
    inference_type="sampler"
) :
    model_output = generate_residuals_norm(
        model,
        features,
        labels,
        targets,
        loss_fn,
        args,
        device,
        logger,
        epoch,
        batch_idx,
        inference_type,
    )
    return dataset.model_output_to_pred(model_output, coarses)

def run_validation(
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
    writer=None,
    plot_every_n_epochs=None,
    paths=None,
    compute_crps=False,
    crps_batch_size=2,
    crps_ensemble_size=10,
    save_predicted = False,
    save_reference = False,
    filename_suffix = None
):
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
    for k in args.varnames_list :
        for m in deterministic_metrics:
            val_metrics[f"{k}_pred_vs_fine_{m}"] = (
                MetricTracker()
            )  # Model prediction vs true fine
            val_metrics[f"{k}_coarse_vs_fine_{m}"] = (
                MetricTracker()
            )  # Coarse vs true fine (baseline)
        if compute_crps:
            val_metrics[f"{k}_pred_vs_fine_CRPS"] = MetricTracker()
    for k in args.diagnostics_varnames_list :
        for m in deterministic_metrics:
            val_metrics[f"{k}_pred_vs_fine_{m}"] = (
                MetricTracker()
            )  # Model prediction vs true fine
        if compute_crps:
            val_metrics[f"{k}_pred_vs_fine_CRPS"] = MetricTracker()

    # Add average metrics across all variables for each metric type
    for m in deterministic_metrics:
        val_metrics[f"average_pred_vs_fine_{m}"] = MetricTracker()
        val_metrics[f"average_coarse_vs_fine_{m}"] = MetricTracker()
    if compute_crps:
        val_metrics["average_pred_vs_fine_CRPS"] = MetricTracker()

    all_data = {"predictions": [], "coarse": [], "fine": [], "diagnostic": [], "latitude_fine": [], "longitude_fine": [], "times": []}

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
            # only run for the first 50 batches during training to save compute time
            if args.run_type != "inference" and batch_idx == 50 :
                break
            # Move data to device
            features = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            coarse = batch["coarse"].to(device)
            # Number of variables (channels)
            n_vars = len(args.varnames_list)
            n_diagnostic_vars = len(args.diagnostics_varnames_list)
            fine = batch["fine"].to(device)
            diagnostic = batch["diagnostics"].to(device)
            lat_batch = batch["corrdinates"]["lat"].to(device)
            lon_batch = batch["corrdinates"]["lon"].to(device)
            time_index_batch = batch["time_index"]
            if epoch == 0 and batch_idx == 0:
                logger.info(
                    f"Validation batch idx:{batch_idx}\n"
                    f"features shape:{features.shape}, targets shape:{targets.shape}\n"
                    f"coarse shape:{coarse.shape}, fine shape:{fine.shape}\n"
                    f"diagnostics shape: {diagnostic.shape}\n"
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

            final_prediction_batch = generate(
                model=model,
                dataset = valid_dataset,
                features=features,
                coarses=coarse,
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

            for var_name in args.varnames_list:
                # Get the correct channel index for this variable
                iv = index_mapping[var_name]
                fine_var = fine[:, iv : iv + 1]
                coarse_var = coarse[:, iv : iv + 1]
                final_prediction = final_prediction_batch[:,iv : iv + 1,:,:]
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
            for var_name in args.diagnostics_varnames_list :
                # Get the correct channel index for this variable
                iv = index_mapping_diagnostics[var_name]
                diagnostic_var = diagnostic[:, iv : iv + 1]
                final_prediction = final_prediction_batch[:,n_vars + iv : n_vars + iv + 1,:,:]
                # Calculate all metrics for this variable
                for metric_name in deterministic_metrics:
                    metric_func = metric_funcs[metric_name]
                    # Model prediction vs fine
                    num_elements_pred, metric_value_pred = metric_func(
                        final_prediction, diagnostic_var
                    )
                    val_metrics[f"{var_name}_pred_vs_fine_{metric_name}"].update(
                        metric_value_pred.item(), num_elements_pred
                    )
                    # Accumulate for batch averages
                    batch_metric_sums[metric_name]["pred"].update(
                        metric_value_pred.item(), num_elements_pred
                    )
            # Store only needed data for reconstruction
            # Validation outputs are accumulated and immediately moved to CPU
            # to avoid CUDA out-of-memory errors.
            # TODO change if we run out of RAM
            all_data["predictions"].append(final_prediction_batch.detach().cpu())
            all_data["coarse"].append(coarse.detach().cpu())
            all_data["fine"].append(fine.detach().cpu())
            all_data["diagnostic"].append(diagnostic.detach().cpu())
            all_data["latitude_fine"].append(lat_batch.detach().cpu())  # [B, H]
            all_data["longitude_fine"].append(lon_batch.detach().cpu())  # [B, W]
            all_data["times"].append(  # [B]
                np.array([valid_dataset.time.values[time_index] for time_index in time_index_batch])
            )
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
    if save_predicted :
        save_generated_data(
            all_data,
            args.varnames_list + args.diagnostics_varnames_list,
            logger,
            filename_suffix,
            save_dir = paths.results,
            save_reference = save_reference
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
                final_prediction_batch = generate(
                    model=model,
                    dataset = valid_dataset,
                    features=features,
                    coarses=batch["coarse"].to(device),
                    labels=labels,
                    targets=batch["targets"].to(device),
                    loss_fn=loss_fn,
                    args=args,
                    device=device,
                    logger=logger,
                    epoch=epoch,
                    batch_idx=-1, #not tied to validation loop
                    inference_type=args.inference_type,
                )

                ens_preds.append(final_prediction_batch)

            # Compute CRPS per variable
            pred_ens = torch.stack(ens_preds, dim=0)  # [N_ens, B, C, H, W]

            for var_name in args.diagnostics_varnames_list :
                iv = index_mapping_diagnostics[var_name]

                pred_ens_var = pred_ens[:, :, n_vars + iv : n_vars + iv + 1, :, :]  # [N_ens, B, 1, H, W]
                diagnostic_var = batch["diagnostics"][:, iv : iv + 1].to(device)

                pred_ens_flat = pred_ens_var.reshape(crps_ensemble_size, -1)
                true_flat = diagnostic_var.reshape(-1)

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
    for var_name in args.diagnostics_varnames_list:
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
                logger.info(f"   └── {metric_name}:")
                logger.info(
                    f"       └── Model Pred vs Fine: {pred_metric:.4f} ± {pred_std:.4f}"
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
        try:
            heatmap_path = plot_metrics_heatmap(
                valid_metrics_history=val_metrics,
                variable_names=args.diagnostics_varnames_list,
                metric_names=metric_names,
                filename=f"{args.run_type}_validation_metrics_diag_epoch_{epoch}",
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
        for var_name in args.diagnostics_varnames_list:
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

    return avg_val_loss, val_metrics