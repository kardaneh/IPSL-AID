import sys
sys.path.insert(1,"/lustre/fswork/projects/rech/huw/uzb76ep/IPSL_AID/IPSL-AID/")

import os
import numpy as np
import time
import matplotlib.pyplot as plt
import cartopy
import xarray as xr
import torchvision
from torchvision import transforms
import torch
import torch.nn.functional as F
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import datetime
from scipy import stats
import logging
import argparse
import scipy as sp
import mpltex
import IPSL_AID.diagnostics

def RMSE(generated, reference, pixel_wise = True) :
    """
    generated and reference are 5D tensors of shape (M,T, C, h, w)
    M nb members, T nb of timesteps, C nb of channels, (h, w) resolution of geographical domain.
    """
    diff = (generated - reference)**2
    if pixel_wise :
        return torch.sqrt(torch.mean(diff, dim=(0,1)))
    else :
        return torch.sqrt(torch.mean(diff, dim = (0,1,3,4)))
  
def MAE(generated, reference, pixel_wise = True) :
    """
    generated and reference are 5D tensors of shape (M, T, C, h, w)
    T nb of timesteps, (h, w) resolution of geographical domain.
    """
    diff = torch.abs(generated - reference)
    if pixel_wise :
        return torch.mean(diff, dim=(0,1))
    else :
        return torch.mean(diff, dim = (0,1,3,4))

def bias_(generated, reference, pixel_wise = True) :
    """
    generated and reference are 5D tensors of shape (M, T, C, h, w)
    T nb of timesteps, (h, w) resolution of geographical domain.
    """
    diff = (generated - reference)
    if pixel_wise :
        return torch.mean(diff, dim=(0,1))
    else :
        return torch.mean(diff, dim = (0,1,3,4))

def CRPS(generated_ens, reference) :
    # generated_ens of shape (nb_ens,T,C,L,W)
    # reference of shape (T,C,L,W)
    nb_ens,T,C,L,W = generated_ens.shape
    #reference = reference.reshape(1,L,W)
    bias_term = torch.mean(torch.abs(generated_ens - reference), dim=0)
    diversity_term = torch.zeros_like(reference)
    count = 0
    for i in range(nb_ens) :
        for j in range(nb_ens) :
            diversity_term += torch.abs(generated_ens[i] - generated_ens[j])
            count += 1
    diversity_term = diversity_term /(2*count)
    return torch.mean(bias_term - diversity_term, dim=0)

def plot_metric(
    metric_tensor,
    metric_name,
    longitude,
    latitude,
    channel_names = ["precipitations"],
    baselines = [],
    baseline_names = [],
    units = ["mm/6h"],
    save_path = None,
    **kwargs
    ) :
    size_subplot = 6
    # metric tensor has to be of shape (C,h,w)
    # plot the bias map between generated and reference
    nb_channels = metric_tensor.shape[0]
    if save_path is None :
        save_path = f"./results/{metric_name}.png"
    if nb_channels == 1 :
        fig,ax = plt.subplots(nb_channels,len(baselines)+1,subplot_kw={"projection": ccrs.PlateCarree()}, figsize=((len(baselines)+1)*size_subplot, nb_channels * size_subplot) )
        ax[0].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        ax[0].coastlines(resolution='50m', alpha = 0.5)
        ax[0].set_axis_off()
        ax[0].set_title(metric_name + " for " + channel_names[0])
        img0 = ax[0].pcolormesh(
            longitude,
            latitude,
            metric_tensor[0],
            transform= ccrs.PlateCarree(),
            **kwargs
        )
        for i,baseline in enumerate(baselines) :
            ax[i+1].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
            ax[i+1].coastlines(resolution='50m', alpha = 0.5)
            ax[i+1].set_axis_off()
            ax[i+1].set_title(metric_name + " for " + channel_names[0] + " - " + baseline_names[i])
            img0 = ax[i+1].pcolormesh(
                longitude,
                latitude,
                baseline[0],
                transform= ccrs.PlateCarree(),
                **kwargs
            )
        fig.colorbar(img0,ax = ax, label = units[0], location = "right")
        fig.savefig(save_path)
        plt.close()
    if nb_channels > 1 :
        fig,ax = plt.subplots(nb_channels,len(baselines)+1,subplot_kw={"projection": ccrs.PlateCarree()}, figsize=((len(baselines)+1)*size_subplot, nb_channels * size_subplot))
        for i in range(nb_channels) :
            ax[i,0].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
            ax[i,0].coastlines(resolution='50m', alpha = 0.5)
            ax[i,0].set_axis_off()
            ax[i,0].set_title(metric_name + " for " + channel_names[i])
            img0 = ax[i,0].pcolormesh(
                longitude,
                latitude,
                metric_tensor[i],
                transform= ccrs.PlateCarree(),
                **kwargs
            )
            for j,baseline in enumerate(baselines) :
                ax[i,j+1].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
                ax[i,j+1].coastlines(resolution='50m', alpha = 0.5)
                ax[i,j+1].set_axis_off()
                ax[i,j+1].set_title(metric_name + " for " + channel_names[i] + " - " + baseline_names[j])
                img0 = ax[i,j+1].pcolormesh(
                    longitude,
                    latitude,
                    baseline[i],
                    transform= ccrs.PlateCarree(),
                    **kwargs
                )
            fig.colorbar(img0,ax = ax[i], label = units[i], location = "right")
        fig.savefig(save_path)
        plt.close()

def zero_proportion_bias(generated, reference, channel = -1, pixel_wise = True, threshold = 0.1) :
    """
    computes the difference between the proportions of zeros of the generated and reference distributions of precipitations.
    generated is 5D tensors of shape (M, T, C, h, w)
    reference is 4D tensors of shape (T, C, h, w)
    T nb of timesteps, (h, w) resolution of geographical domain.
    """
    mask_generated = (generated[:,:,channel,:,:] < threshold).to(torch.float32)
    mask_reference = (reference[:,channel,:,:] < threshold).to(torch.float32)
    if pixel_wise :
        return torch.mean(mask_generated, dim=(0,1)) - torch.mean(mask_reference, dim=0)
    return torch.mean(mask_generated) - torch.mean(mask_reference)

def zero_proportion(tensor, channel = -1, pixel_wise = True, threshold = 0.1) :
    """
    computes the difference between the proportions of zeros of the generated and reference distributions of precipitations.
    generated is 5D tensors of shape (M, T, C, h, w)
    reference is 4D tensors of shape (T, C, h, w)
    T nb of timesteps, (h, w) resolution of geographical domain.
    """
    if len(tensor.shape) == 5 :
        mask_tensor = (tensor[:,:,channel,:,:] < threshold).to(torch.float32)
        if pixel_wise :
            return torch.mean(mask_tensor, dim=(0,1))
        return torch.mean(mask_tensor)
    else :
        mask_tensor = (tensor[:,channel,:,:] < threshold).to(torch.float32)
        if pixel_wise :
            return torch.mean(mask_tensor, dim=0)
        return torch.mean(mask_tensor)

def prepare_RAPSD(tensor, norm = True, channel_rain = -1, threshold_rain = None) :
    """
    does all the work for one tensor to help plot its RAPSD.
    """
    if len(tensor.shape) == 5 :
        M,T,C,h,w = tensor.shape
        tensor = tensor.reshape((M*T,C,h,w))
    if norm :
        std_tensor = torch.std(tensor, dim=(0,2,3))[None,:,None,None]
        tensor = tensor / std_tensor        
    if threshold_rain is not None :
        mask_tensor = (tensor[:,channel_rain,:,:] >= threshold_rain)
        tensor[:,channel_rain,:,:] = torch.where(mask_tensor, tensor[:,channel_rain,:,:], 0)
    return tensor

def compute_spectrum2d(data, resolution = 0.25): #code from inference folder (Watts)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    N, N_y, N_x = data.shape
    if N_x == 2 * N_y:
        data1 = data[:, :, :N_y]
        data2 = data[:, :, N_y:]
        data = np.concatenate((data1, data2), axis=0)
    N, N_y, N_x = data.shape

    # Take FFT and take amplitude
    fourier_image = np.fft.fftn(data, axes=(1, 2))
    fourier_amplitudes = np.abs(fourier_image) ** 2

    # Get kx and ky
    kfreq_x = np.fft.fftfreq(N_x,d=resolution)
    kfreq_y = np.fft.fftfreq(N_y,d=resolution)

    # Combine into one wavenumber for both directions
    kfreq2D = np.meshgrid(kfreq_x, kfreq_y)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = np.repeat(knrm[np.newaxis, ...], repeats=N, axis=0)

    # Flatten arrays
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    # Get k-bins and mean amplitude within each bin
    kbins = np.linspace(start=0, stop=1/(2*resolution), num=N_x//2)
    Abins, _, _ = stats.binned_statistic(
        knrm, fourier_amplitudes, statistic="mean", bins=kbins
    )

    # Multiply by volume of bin
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)

    # Get center of k-bin for plotting
    kvals = 0.5 * (kbins[1:] + kbins[:-1])

    return (kvals, Abins)

def plot_RAPSD(
        generated,
        reference,
        baselines = [],
        baseline_names = [],
        reference_coarse = None,
        norm = True,
        channel_rain = -1,
        threshold_rain = None,
        variables=["precipitations"],
        save_path = "./results/RAPSD.png",
        figsize_multiplier = 4,
    ) :
    # baselines is a list of tensors of same shape as generated
    M,T,C,h,w = generated.shape
    data_gen = prepare_RAPSD(
        tensor = generated,
        norm = norm,
        channel_rain=channel_rain,
        threshold_rain = threshold_rain
    )
    data_ref = prepare_RAPSD(
        tensor = reference,
        norm = norm,
        channel_rain=channel_rain,
        threshold_rain = threshold_rain
    )
    if reference_coarse is not None :
        data_ref_coarse = prepare_RAPSD(
            tensor = reference_coarse,
            norm = norm,
            channel_rain=channel_rain,
            threshold_rain = threshold_rain
        )
    if len(baselines) != 0 :
        data_baselines = []
        for baseline in baselines :
            data_baseline = prepare_RAPSD(
                tensor = baseline,
                norm = norm,
                channel_rain=channel_rain,
                threshold_rain = threshold_rain
            )
            data_baselines.append(data_baseline)

    fig, axes = plt.subplots(
        1,
        len(variables),
        figsize=(len(variables) * figsize_multiplier, figsize_multiplier),
        squeeze=False,
    )  # nrows * figsize_multiplier
    plt.subplots_adjust(
        hspace=0.2, wspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )
    axes = axes.ravel()
    for ax in axes:
        ax.set_box_aspect(1)

    if C == 1 :
        linestyles = mpltex.linestyle_generator(markers=[]) 
        ax_bottom = axes[0]
        ax_bottom.set_title(f"{variables[0]}")
        kvals, gen_bins = compute_spectrum2d(data = data_gen[:,0,:,:])
        _, ref_bins = compute_spectrum2d(data = data_ref[:,0,:,:])
        # Plot all spectra
        ax_bottom.loglog(kvals, gen_bins, label="generated", **next(linestyles))
        ax_bottom.loglog(kvals, ref_bins, label="ERA5", **next(linestyles))
        if len(baselines) != 0 :
            for i,data_baseline in enumerate(data_baselines) :
                _, data_baseline_bins = compute_spectrum2d(data = data_baseline[:,0,:,:])
                ax_bottom.loglog(kvals, data_baseline_bins, label=baseline_names[i], **next(linestyles))
        if reference_coarse is not None :
            _, ref_coarse_bins = compute_spectrum2d(data = data_ref_coarse[:,0,:,:])
            ax_bottom.loglog(kvals, ref_coarse_bins, label="LMDz", **next(linestyles))

        ax_bottom.set_ylabel("PSD(k)")

        # Always show x-axis label
        ax_bottom.set_xlabel("Wavenumber k [cycles/°]")

        ax_bottom.legend()
        ax_bottom.grid(True, alpha=0.3, which="both")

        # Set reasonable axis limits
        valid = (kvals > 0) & (ref_bins > 0)
        if np.any(valid):
            ax_bottom.set_xlim(kvals[valid][0] * 0.8, kvals[valid][-1] * 1.2)

            # Find y-range
            y_min = min(gen_bins[valid].min(), ref_bins[valid].min())
            y_max = max(gen_bins[valid].max(), ref_bins[valid].max())
            if reference_coarse is not None:
                y_min = min(y_min, ref_coarse_bins[valid].min())
                y_max = max(y_max, ref_coarse_bins[valid].max())

            ax_bottom.set_ylim(y_min * 0.5, y_max * 2.0)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        fig.savefig(save_path)
        plt.close()
        return None

    for i in range(C) : 
        linestyles = mpltex.linestyle_generator(markers=[]) 
        ax_bottom = axes[i]
        ax_bottom.set_title(f"{variables[i]}") 
        kvals, gen_bins = compute_spectrum2d(data = data_gen[:,i,:,:])
        _, ref_bins = compute_spectrum2d(data = data_ref[:,i,:,:])
        # Plot all spectra
        ax_bottom.loglog(kvals, gen_bins, label="generated", **next(linestyles))
        ax_bottom.loglog(kvals, ref_bins, label="ERA5", **next(linestyles))
        #     ax[i].plot(kvals, np.log10(ref_coarse_bins), c = "black", label = "LMDZ")
        # ax[i].plot(kvals, np.log10(gen_bins), c = "red", label = "generated")
        # ax[i].plot(kvals, np.log10(ref_bins), c = "blue", label = "ERA5")
        if len(baselines) != 0 :
            for j,data_baseline in enumerate(data_baselines) :
                _, data_baseline_bins = compute_spectrum2d(data = data_baseline[:,i,:,:])
                ax_bottom.loglog(kvals, data_baseline_bins, label=baseline_names[j], **next(linestyles))
                #ax[i].plot(kvals, np.log10(data_baseline_bins), label = baseline_names[j])
        if reference_coarse is not None :
            _, ref_coarse_bins = compute_spectrum2d(data = data_ref_coarse[:,i,:,:])
            ax_bottom.loglog(kvals, ref_coarse_bins, label="LMDz", **next(linestyles))
            
        # ax[i].set_title(f"Radially averaged power spectral density of {variables[i]}")
        # ax[i].set_xlabel("k (km-1)")
        # ax[i].set_ylabel("log(P(k))")
        # ax[i].legend()

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
        valid = (kvals > 0) & (ref_bins > 0)
        if np.any(valid):
            ax_bottom.set_xlim(kvals[valid][0] * 0.8, kvals[valid][-1] * 1.2)

            # Find y-range
            y_min = min(gen_bins[valid].min(), ref_bins[valid].min())
            y_max = max(gen_bins[valid].max(), ref_bins[valid].max())
            if reference_coarse is not None:
                y_min = min(y_min, ref_coarse_bins[valid].min())
                y_max = max(y_max, ref_coarse_bins[valid].max())

            ax_bottom.set_ylim(y_min * 0.5, y_max * 2.0)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    fig.savefig(save_path)
    plt.close()
    return None

def rain_threshold(rain_tensor, threshold = 1e-2) :
    mask = (rain_tensor >= threshold)
    rain_tensor = torch.where(mask, rain_tensor, 0)
    return rain_tensor

# TODO adapt or remove, this function is directly from IPSL-AID, if adapted we can pull request into IPSL-AID.
# there seems to be a bug if we have two baselines or more, ?
def plot_validation_pdfs(
    predictions,  # Model predictions (fine predicted)
    targets,  # Ground truth (fine true)
    coarse_inputs=None,  # Coarse inputs for comparison (optional)
    baselines=[],  # Baseline inputs for comparison (optional)
    baseline_names=[],  # Baseline names (optional)
    variable_names=None,  # List of variable names
    units = None, #list of variable units
    filename="validation_pdfs.png",
    save_dir="./results",
    log_scale=False,
    low_lim_percentile = None, # percentile to use for lower limit of x axis
    high_lim_percentile = None, # percentile to use for higher limit of x axis
    figsize_multiplier=4,  # Base size per subplot
    save_npz=False,
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
    if isinstance(log_scale, bool) :
        log_scale = [log_scale for var in variable_names]

    # Convert to numpy if they're tensors
    if hasattr(predictions, "detach"):
        predictions = predictions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()
    if coarse_inputs is not None and hasattr(coarse_inputs, "detach"):
        coarse_inputs = coarse_inputs.detach().cpu().numpy()
    baseline_outputs = []
    for baseline_output in baselines :
        if hasattr(baseline_output, "detach") :
            baseline_output = baseline_output.detach().cpu().numpy()
        baseline_outputs.append(baseline_output)

    batch_size, num_vars, h, w = predictions.shape

    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f"Variable {i + 1}" for i in range(num_vars)]

    plot_variable_names = [IPSL_AID.diagnostics.PlotConfig.get_plot_name(var) for var in variable_names]

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
        pred_i = IPSL_AID.diagnostics.PlotConfig.convert_units(var_name, predictions[:, i])
        tgt_i = IPSL_AID.diagnostics.PlotConfig.convert_units(var_name, targets[:, i])
        # plot_name = plot_variable_names[i]
        plot_name = var_name + f" [{units[i]}]"

        pred_flat = pred_i.reshape(-1)
        target_flat = tgt_i.reshape(-1)
        # convert units and flatten for baselines
        baseline_outputs_flat = [
            IPSL_AID.diagnostics.PlotConfig.convert_units(var_name, baseline[:, i]).reshape(-1)
            for baseline in baseline_outputs
        ]            

        # Collect all data for combined range
        all_data = [pred_flat, target_flat] + baseline_outputs_flat
        if coarse_inputs is not None:
            # coarse_flat = coarse_inputs[:, i, :, :].flatten() #.mean(axis=0).reshape(-1)
            coarse_i = IPSL_AID.diagnostics.PlotConfig.convert_units(var_name, coarse_inputs[:, i])
            coarse_flat = coarse_i.reshape(-1)

            all_data.append(coarse_flat)

        # Calculate global range for consistent x-axis
        all_values = np.concatenate(all_data)
        if low_lim_percentile is None :
            data_min = np.min(all_values)  # 0.5th percentile # min
        else :
            data_min = np.percentile(all_values, low_lim_percentile)
        if high_lim_percentile is None : 
            data_max = np.max(all_values)  # 99.5th percentile # max
        else :
            data_max = np.percentile(all_values, high_lim_percentile)
        data_range = data_max - data_min

        # Extend range slightly for better visualization
        x_min = data_min #- 0.05 * data_range
        x_max = data_max #+ 0.05 * data_range

        # Create bins for PDF calculation
        n_bins = 100
        bins = np.linspace(x_min, x_max, n_bins + 1)

        # Small epsilon to avoid log(0)
        epsilon = 1e-12

        # Plot log PDFs
        # Plot predictions
        hist_pred, bin_edges = np.histogram(pred_flat, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if log_scale[i] :
            log_hist_pred = np.log10(hist_pred + epsilon)
            ax.plot(bin_centers, log_hist_pred, label="Generated", **next(linestyles))
        else :
            ax.plot(bin_centers, hist_pred, label="Generated", **next(linestyles))

        # Plot ground truth
        hist_target, _ = np.histogram(target_flat, bins=bins, density=True)
        if log_scale[i]:
            log_hist_target = np.log10(hist_target + epsilon)
            ax.plot(bin_centers, log_hist_target, label="ERA5", **next(linestyles))
        else :
            ax.plot(bin_centers, hist_target, label="ERA5", **next(linestyles))

        #Plot baselines
        for baseline_output_flat, baseline_name in zip(baseline_outputs_flat, baseline_names) :
            hist_baseline, _ = np.histogram(baseline_output_flat, bins=bins, density=True)
            if log_scale[i] :
                log_hist_baseline = np.log10(hist_baseline + epsilon)
                ax.plot(bin_centers, log_hist_baseline, label=baseline_name, **next(linestyles))
            else :
                ax.plot(bin_centers, hist_baseline, label=baseline_name, **next(linestyles))

        # Plot coarse inputs if available
        if coarse_inputs is not None:
            hist_coarse, _ = np.histogram(coarse_flat, bins=bins, density=True)
            if log_scale[i] :
                log_hist_coarse = np.log10(hist_coarse + epsilon)
                ax.plot(bin_centers, log_hist_coarse, label="LMDz", **next(linestyles))
            else :
                ax.plot(bin_centers, hist_coarse, label="LMDz", **next(linestyles))
    
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

        for baseline_output_flat, baseline_name in zip(baseline_outputs_flat, baseline_names) :
            baseline_mean = np.mean(baseline_output_flat)
            baseline_std = np.std(baseline_output_flat)
            stats_text.append(f"{baseline_name}: μ={baseline_mean:.3f}, σ={baseline_std:.3f}")

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
        # REMOVED because it bugs when having ensemble pred to fix
        # correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
        # stats_text.append(f"Correlation: {correlation:.4f}")

        # Log statistics instead of plotting them
        print(f"[PDF stats] {plot_name}")
        print(f"  Predictions: μ={pred_mean:.3f}, σ={pred_std:.3f}")
        print(f"  Ground Truth: μ={target_mean:.3f}, σ={target_std:.3f}")
        if coarse_inputs is not None:
            print(f"  Coarse: μ={coarse_mean:.3f}, σ={coarse_std:.3f}")
        print(f"  KL Divergence: {kl_divergence:.4f}")
        # print(f"  Correlation: {correlation:.4f}")

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
        if log_scale[i] :
            y_min = min(log_hist_pred.min(), log_hist_target.min())
            if coarse_inputs is not None:
                y_min = min(y_min, log_hist_coarse.min())
            y_max = max(log_hist_pred.max(), log_hist_target.max())
            if coarse_inputs is not None:
                y_max = max(y_max, log_hist_coarse.max())
        else :
            y_min = min(hist_pred.min(), hist_target.min())
            if coarse_inputs is not None:
                y_min = min(y_min, hist_coarse.min())
            y_max = max(hist_pred.max(), hist_target.max())
            if coarse_inputs is not None:
                y_max = max(y_max, hist_coarse.max())  

        # Add small margin to y-limits
        y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # Use scientific notation for large ranges
        if data_range > 1000:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        if save_npz:
            key = f"{var_name}__pdf__"

            pdf_npz_data[key + "bin_centers"] = bin_centers
            if log_scale[i]:
                pdf_npz_data[key + "log_pred"] = log_hist_pred
                pdf_npz_data[key + "log_truth"] = log_hist_target
            else :
                pdf_npz_data[key + "pred"] = hist_pred
                pdf_npz_data[key + "truth"] = hist_target

            pdf_npz_data[key + "mean_pred"] = pred_mean
            pdf_npz_data[key + "std_pred"] = pred_std
            pdf_npz_data[key + "mean_truth"] = target_mean
            pdf_npz_data[key + "std_truth"] = target_std
            pdf_npz_data[key + "kl"] = kl_divergence
            #pdf_npz_data[key + "corr"] = correlation

            if coarse_inputs is not None:
                if log_scale[i] :
                    pdf_npz_data[key + "log_coarse"] = log_hist_coarse
                else :
                    pdf_npz_data[key + "coarse"] = hist_coarse
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


def get_divergence_gen_ref(generated,reference, channel_u, channel_v) :
    gen_u = generated[:,:,channel_u,:,:]
    gen_v = generated[:,:,channel_v,:,:]
    ref_u = reference[:,channel_u,:,:]
    ref_v = reference[:,channel_v,:,:]
    div_gen = divergence(gen_u,gen_v)
    div_ref = divergence(ref_u,ref_v)
    return div_gen, div_ref

def get_divergence(tensor, channel_u, channel_v):
    u = tensor[:,channel_u,:,:]
    v = tensor[:,channel_v,:,:]
    div= divergence(u,v)
    return div
    
def get_curl_gen_ref(generated,reference, channel_u, channel_v) :
    gen_u = generated[:,:,channel_u,:,:]
    gen_v = generated[:,:,channel_v,:,:]
    ref_u = reference[:,channel_u,:,:]
    ref_v = reference[:,channel_v,:,:]
    curl_gen = curl(gen_u,gen_v)
    curl_ref = curl(ref_u,ref_v)
    return curl_gen, curl_ref

def get_curl(tensor, channel_u, channel_v) :
    u = tensor[:,channel_u,:,:]
    v = tensor[:,channel_v,:,:]
    curl_ = curl(u,v)
    return curl_    

def get_windspeed(u,v) :
    return torch.sqrt(u**2 + v**2)

def nb_extremes_above_threshold(
    generated,
    fine,
    lon,
    lat,
    variable_name,
    threshold,
    unit,
    save_path,
    coarse = None,
    baselines = [],
    baseline_names = [],
    fontsize_towns = 14,
    fontsize_ticks = 25,
):
    size_subplot = 6
    # generated shape B,T,H,W
    freq_extremes_generated = torch.mean((generated > threshold).float(), dim=(0,1))
    if coarse is not None :
        freq_extremes_coarse = torch.mean((coarse > threshold).float() , dim=0)
    freq_extremes_fine = torch.mean((fine > threshold).float() , dim=0)
    baseline_freqs = []
    for baseline in baselines :
        freq_extremes_baseline = torch.mean((baseline > threshold).float() , dim=(0,1))
        baseline_freqs.append(freq_extremes_baseline)

    vmin = 0
    vmax = max(
        torch.max(freq_extremes_fine),
        torch.max(freq_extremes_generated),
        *[torch.max(freq_extremes_baseline) for freq_extremes_baseline in baseline_freqs],
    )
    if coarse is not None :
        vmax = max(vmax, torch.max(freq_extremes_coarse))
        fig,ax = plt.subplots(1,len(baselines)+3, subplot_kw={"projection": ccrs.PlateCarree()}, figsize = ((len(baselines)+3)*size_subplot, size_subplot))
    else : 
        fig,ax = plt.subplots(1,len(baselines)+2, subplot_kw={"projection": ccrs.PlateCarree()}, figsize = ((len(baselines)+2)*size_subplot, size_subplot))

    if coarse is not None :
        ax[-1].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        ax[-1].coastlines(resolution='50m', alpha = 0.5)
        ax[-1].set_axis_off()
        img0 = ax[-1].pcolormesh(
            lon,
            lat,
            freq_extremes_coarse,
            transform= ccrs.PlateCarree(),
            cmap = "Oranges",
            vmin = vmin,
            vmax = vmax
        )
        ax[-1].set_title(f"LMDZ")

    ax[0].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    ax[0].coastlines(resolution='50m', alpha = 0.5)
    ax[0].set_axis_off()
    img0 = ax[0].pcolormesh(
        lon,
        lat,
        freq_extremes_fine,
        transform= ccrs.PlateCarree(),
        cmap = "Oranges",
        vmin = vmin,
        vmax = vmax
    )
    ax[0].set_title(f"ERA5")

    ax[1].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
    ax[1].coastlines(resolution='50m', alpha = 0.5)
    ax[1].set_axis_off()
    img0 = ax[1].pcolormesh(
        lon,
        lat,
        freq_extremes_generated,
        transform= ccrs.PlateCarree(),
        cmap = "Oranges",
        vmin = vmin,
        vmax = vmax
    )
    ax[1].set_title(f"diffusion")

    for i,baseline_freq in enumerate(baseline_freqs) :
        ax[i+2].add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5)
        ax[i+2].coastlines(resolution='50m', alpha = 0.5)
        ax[i+2].set_axis_off()
        img1 = ax[i+2].pcolormesh(
            lon,
            lat,
            baseline_freq,
            transform= ccrs.PlateCarree(),
            cmap = "Oranges",
            vmin = vmin,
            vmax = vmax
        )
        ax[i+2].set_title(f"{baseline_names[i]}")
    
    cbar = fig.colorbar(img0,ax = ax, label = "frequency", location = "left")
    cbar.ax.tick_params(labelsize=fontsize_ticks)
    cbar.set_label('frequency', size=fontsize_ticks)
    fig.savefig(save_path)
    fig.suptitle(f"Frequency of {variable_name} above {threshold} {unit}")
    plt.close()

def main_full_vars(
    path_folder = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDZ-ERA/test_model_14_05_2026/results/",
    pattern_data_files = "test_save_validation",
    nb_members = 10,
    path_log = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDZ-ERA/test_model_14_05_2026/results/",
    save_path = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDZ-ERA/test_model_14_05_2026/results/validation/",
    variable_names = ["u850", "v850", "t850", "TCWV", "CAPE", "precip"],
    variable_units = ["m/s", "m/s", "K", "mm", "J/kg", "mm/6h"],
    path_folder_Unet = "/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDZ-ERA/unet_baseline_04_06_2026/results/",
    convert_to_mm = True,
    convert_baseline_to_mm = True
) :
    # create logger :
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename = path_log + f'log_validate.log', level=logging.INFO)
    logger.info('Initializing the job')
    logger.info(f'validating data here : {path_folder}')
    # get longitude and latitude :
    gen_i_xr = xr.open_dataset(path_folder + f"predictions_{pattern_data_files}_member_0.nc")
    if "longitude" in list(gen_i_xr.coords) :
        longitude = gen_i_xr["longitude"].values
        latitude = gen_i_xr["latitude"].values
    else :
        longitude = gen_i_xr["lon"].values
        latitude = gen_i_xr["lat"].values

    n_vars = len(variable_names)

    #import generated and reference data :
    generated_list = []
    for i in range(nb_members) :
        gen_i_numpy = np.load(path_folder + f"predictions_{pattern_data_files}_member_{i}.npy")
        gen_i = torch.from_numpy(gen_i_numpy)
        gen_i = gen_i[None,...]
        generated_list.append(gen_i)
    generated = torch.cat(generated_list, dim = 0)
    generated[:,:,5,:,:] = torch.clamp(generated[:,:,5,:,:], min=0) # clamp to 0 to avoid negative values
    if convert_to_mm :
        generated[:,:,5,:,:] = generated[:,:,5,:,:] * 1000 # convert ERA5 precip to mm/6h
    # load coarses :
    # reference_coarse_np = np.load(path_folder + f"coarses_{pattern_data_files}_member_0.npy")
    # reference_coarse = torch.from_numpy(reference_coarse_np)

    # load coarses by loading the coarses of UNet:
    reference_coarse_np = np.load(path_folder_Unet + f"coarses_{pattern_data_files}_member_0.npy")
    reference_coarse = torch.from_numpy(reference_coarse_np)
    print(f"reference coarse shape : {reference_coarse.shape}")

    # load fine :
    reference_fine_np = np.load(path_folder + f"fines_{pattern_data_files}_member_0.npy")
    reference_fine = torch.from_numpy(reference_fine_np)
    if convert_to_mm :
        reference_fine[:,5,:,:] = reference_fine[:,5,:,:]*1000

    baselines = []
    baseline_names = ['baseline_unet']
    baseline_unet = np.load(path_folder_Unet + f"predictions_{pattern_data_files}_member_0.npy")
    baseline_unet = torch.from_numpy(baseline_unet)
    baseline_unet = baseline_unet[None,...]
    baseline_unet[:,:,5,:,:] = torch.clamp(baseline_unet[:,:,5,:,:], min=0)
    if convert_baseline_to_mm :
        baseline_unet[:,:,5,:,:] = baseline_unet[:,:,5,:,:]*1000
    baselines.append(baseline_unet)

    print(f"Generated shape : {generated.shape}")
    logger.info(f"Generated shape : {generated.shape}")
    print(f"Reference coarse shape : {reference_coarse.shape}")
    logger.info(f"Reference coarse shape : {reference_coarse.shape}")
    print(f"Reference fine shape : {reference_fine.shape}")
    logger.info(f"Reference fine shape : {reference_fine.shape}")
    print(f"Baseline unet shape : {baseline_unet.shape}")
    logger.info(f"Baseline unet shape : {baseline_unet.shape}")

    B,T,C,h,w = generated.shape
    generated_all_members = torch.reshape(generated, (B*T,C,h,w))
    generated_ensemble_mean = torch.mean(generated, dim=0)[None,...]

    # SAL inspired characteristics :
    sal_dict_predicted = IPSL_AID.diagnostics.compute_sal_objects_time(
        array = (generated_all_members[:,5,:,:]).cpu().numpy(), 
        eThreshFix=None,  
        thr_quantile=0.95, 
        thr_factor=1/15., 
        minFac=0.1, 
        minsize=0, 
        structure=np.ones((3, 3), dtype=int)
    )
    sal_dict_reference = IPSL_AID.diagnostics.compute_sal_objects_time(
        array = (reference_fine[:,5,:,:]).cpu().numpy(), 
        eThreshFix=None,  
        thr_quantile=0.95, 
        thr_factor=1/15., 
        minFac=0.1, 
        minsize=0, 
        structure=np.ones((3, 3), dtype=int)
    )
    sal_dict_baseline_unet = IPSL_AID.diagnostics.compute_sal_objects_time(
        array = (baseline_unet[0,:,5,:,:]).cpu().numpy(), 
        eThreshFix=None,  
        thr_quantile=0.95, 
        thr_factor=1/15., 
        minFac=0.1, 
        minsize=0, 
        structure=np.ones((3, 3), dtype=int)
    )
    print("computed the SAL characteristics")
    # plot PDFs of characteristics - structure / waVOL :
    structure_predicted = sal_dict_predicted["sal_waVOL"]
    structure_predicted = structure_predicted[~np.isnan(structure_predicted)]
    structure_reference = sal_dict_reference["sal_waVOL"]
    structure_reference = structure_reference[~np.isnan(structure_reference)]
    structure_baseline_unet = sal_dict_baseline_unet["sal_waVOL"]
    structure_baseline_unet = structure_baseline_unet[~np.isnan(structure_baseline_unet)]
    plot_validation_pdfs(
        predictions = structure_predicted[None,None,None,:],
        targets = structure_reference[None,None,None,:],
        variable_names = ["Weighted area volumes component"],
        units = ["no unit"],
        baselines = [structure_baseline_unet[None,None,None,:]],
        baseline_names = ["UNet baseline"],
        low_lim_percentile = 0.1,
        high_lim_percentile = 99.9,
        filename = "pdf_structure.png",
        save_dir = save_path,
        log_scale = True,
    )

    # plot PDFs of characteristics - amplitude :
    amplitude_predicted = sal_dict_predicted["sal_a"]
    amplitude_predicted = amplitude_predicted[~np.isnan(amplitude_predicted)]
    amplitude_reference = sal_dict_reference["sal_a"]
    amplitude_reference = amplitude_reference[~np.isnan(amplitude_reference)]
    amplitude_baseline_unet = sal_dict_baseline_unet["sal_a"]
    amplitude_baseline_unet = amplitude_baseline_unet[~np.isnan(amplitude_baseline_unet)]
    plot_validation_pdfs(
        predictions = amplitude_predicted[None,None,None,:],
        targets = amplitude_reference[None,None,None,:],
        variable_names = ["Amplitude component"],
        units = ["no unit"],
        baselines = [amplitude_baseline_unet[None,None,None,:]],
        baseline_names = ["UNet baseline"],
        low_lim_percentile = 0.1,
        high_lim_percentile = 99.9,
        filename = "pdf_amplitude.png",
        save_dir = save_path,
        log_scale = True,
    )

    # plot PDFs of characteristics - localisation / r :
    r_predicted = sal_dict_predicted["sal_r"]
    r_predicted = r_predicted[~np.isnan(r_predicted)]
    r_reference = sal_dict_reference["sal_r"]
    r_reference = r_reference[~np.isnan(r_reference)]
    r_baseline_unet = sal_dict_baseline_unet["sal_r"]
    r_baseline_unet = r_baseline_unet[~np.isnan(r_baseline_unet)]
    plot_validation_pdfs(
        predictions = r_predicted[None,None,None,:],
        targets = r_reference[None,None,None,:],
        variable_names = ["Weighted average distances"],
        units = ["no unit"],
        baselines = [r_baseline_unet[None,None,None,:]],
        baseline_names = ["UNet baseline"],
        low_lim_percentile = 0.1,
        high_lim_percentile = 99.9,
        filename = "pdf_r.png",
        save_dir = save_path,
        log_scale = True,
    )

    # plot PDFs of characteristics - number of targets :
    targ_num_predicted = sal_dict_predicted["sal_targ_num"]
    targ_num_predicted = targ_num_predicted[~np.isnan(targ_num_predicted)]
    targ_num_reference = sal_dict_reference["sal_targ_num"]
    targ_num_reference = targ_num_reference[~np.isnan(targ_num_reference)]
    targ_num_baseline_unet = sal_dict_baseline_unet["sal_targ_num"]
    targ_num_baseline_unet = targ_num_baseline_unet[~np.isnan(targ_num_baseline_unet)]
    plot_validation_pdfs(
        predictions = targ_num_predicted[None,None,None,:],
        targets = targ_num_reference[None,None,None,:],
        variable_names = ["Number of events"],
        units = ["no unit"],
        baselines = [targ_num_baseline_unet[None,None,None,:]],
        baseline_names = ["UNet baseline"],
        low_lim_percentile = 0.1,
        high_lim_percentile = 99.9,
        filename = "pdf_targ_num.png",
        save_dir = save_path,
        log_scale = True,
    )

    # plot PDFs of characteristics - targ_size :
    targ_size_predicted = sal_dict_predicted["sal_targ_size"]
    targ_size_predicted = targ_size_predicted[~np.isnan(targ_size_predicted)]
    targ_size_reference = sal_dict_reference["sal_targ_size"]
    targ_size_reference = targ_size_reference[~np.isnan(targ_size_reference)]
    targ_size_baseline_unet = sal_dict_baseline_unet["sal_targ_size"]
    targ_size_baseline_unet = targ_size_baseline_unet[~np.isnan(targ_size_baseline_unet)]
    plot_validation_pdfs(
        predictions = targ_size_predicted[None,None,None,:],
        targets = targ_size_reference[None,None,None,:],
        variable_names = ["Event size"],
        units = ["no unit"],
        baselines = [targ_size_baseline_unet[None,None,None,:]],
        baseline_names = ["UNet baseline"],
        low_lim_percentile = 0.1,
        high_lim_percentile = 99.9,
        filename = "pdf_targ_size.png",
        save_dir = save_path,
        log_scale = True,
    )
    print("plotted all SAL characteristics")

    # compute metrics for all the vars :
    rmse_dict = {}
    mae_dict = {}
    bias_dict = {}
    ssr_dict = {}
    bias_value = bias_(generated=generated, reference=reference_fine, pixel_wise=False)
    rmse = RMSE(generated=generated_ensemble_mean, reference=reference_fine, pixel_wise=False)
    mae = MAE(generated=generated_ensemble_mean, reference=reference_fine, pixel_wise=False)
    # compute SSR if ensemble pred :
    if nb_members > 1 :
        ssr_dict = {}
        ssr = IPSL_AID.diagnostics.spread_skill_ratio(predictions = generated, targets=reference_fine, pixel_wise = False, variable_names = variable_names)
    for i, var in enumerate(variable_names) :
        rmse_dict[var] = rmse[i]
        mae_dict[var] = mae[i]
        bias_dict[var] = bias_value[i]
        if nb_members > 1 :  
            ssr_dict[var] = ssr[i]   
    
    zero_proportion_bias_value = zero_proportion_bias(generated, reference_fine, channel=5, pixel_wise=False, threshold=1e-1)
    zero_prop_ref_1e_1 = zero_proportion(reference_fine, channel = 5, pixel_wise=False, threshold=1e-1)
    zero_prop_gen_1e_1 = zero_proportion(generated, channel = 5, pixel_wise=False, threshold=1e-1)
    print(f"RMSE = {rmse_dict}")
    logger.info(f"RMSE = {rmse_dict}")
    print(f"MAE = {mae_dict}")
    logger.info(f"MAE = {mae_dict}")
    print(f"Bias = {bias_dict}")
    logger.info(f"Bias = {bias_dict}")
    if nb_members > 1 :
        print(f"Spread skill ratio = {ssr_dict}")
        logger.info(f"Spread skill ratio = {ssr_dict}")

    print(f"Zero proportion bias for threshold 1e-2 = {zero_proportion_bias_value}")
    logger.info(f"Zero proportion bias for threshold 1e-2 = {zero_proportion_bias_value}")
    print(f"Zero proportion in ERA5 for threshold 1e-1 = {zero_prop_ref_1e_1}")
    print(f"Zero proportion in generated for threshold 1e-1 = {zero_prop_gen_1e_1}")
    logger.info(f"Zero proportion in ERA5 for threshold 1e-1 = {zero_prop_ref_1e_1}")
    logger.info(f"Zero proportion in generated for threshold 1e-1 = {zero_prop_gen_1e_1}")
    rmse_tensor = RMSE(generated=generated_ensemble_mean, reference=reference_fine)
    mae_tensor = MAE(generated=generated_ensemble_mean, reference=reference_fine)
    bias_tensor = bias_(generated=generated, reference=reference_fine)
    zero_proportion_tensor = zero_proportion_bias(generated, reference_fine, channel=5, threshold=1e-1)

    # compare to unet baseline :
    rmse_dict_baseline = {}
    mae_dict_baseline = {}
    bias_dict_baseline = {}
    bias_value_baseline = bias_(generated=baseline_unet, reference=reference_fine, pixel_wise=False)
    rmse = RMSE(generated=baseline_unet, reference=reference_fine, pixel_wise=False)
    mae = MAE(generated=baseline_unet, reference=reference_fine, pixel_wise=False)
    for i, var in enumerate(variable_names) :
        rmse_dict_baseline[var] = rmse[i]
        mae_dict_baseline[var] = mae[i]
        bias_dict_baseline[var] = bias_value_baseline[i]       

    zero_proportion_bias_value = zero_proportion_bias(baseline_unet, reference_fine, channel=5, pixel_wise=False, threshold=1e-1)
    zero_prop_ref_1e_1 = zero_proportion(reference_fine, channel = 5, pixel_wise=False, threshold=1e-1)
    zero_prop_gen_1e_1 = zero_proportion(baseline_unet, channel = 5, pixel_wise=False, threshold=1e-1)
    print(f"RMSE baseline = {rmse_dict_baseline}")
    logger.info(f"RMSE baseline= {rmse_dict_baseline}")
    print(f"MAE baseline= {mae_dict_baseline}")
    logger.info(f"MAE baseline= {mae_dict_baseline}")
    print(f"Bias baseline= {bias_dict_baseline}")
    logger.info(f"Bias baseline= {bias_dict_baseline}")

    print(f"Zero proportion bias for threshold 1e-2 = {zero_proportion_bias_value}")
    logger.info(f"Zero proportion bias for threshold 1e-2 = {zero_proportion_bias_value}")
    print(f"Zero proportion in ERA5 for threshold 1e-1 = {zero_prop_ref_1e_1}")
    print(f"Zero proportion in generated for threshold 1e-1 = {zero_prop_gen_1e_1}")
    logger.info(f"Zero proportion in ERA5 for threshold 1e-1 = {zero_prop_ref_1e_1}")
    logger.info(f"Zero proportion in generated for threshold 1e-1 = {zero_prop_gen_1e_1}")

    # Plot MAE map :
    IPSL_AID.diagnostics.plot_MAE_map(
        predictions = generated_ensemble_mean[0],
        targets = reference_fine,
        lat_1d = latitude,
        lon_1d = longitude,
        variable_names = variable_names,
        save_dir = save_path,
        filename = "MAE_maps.png",
        figsize_multiplier = None,
    )

    # Plot RMSE map :
    print("plotting rmse")
    logger.info("plotting rmse")
    kwargs_rmse = {
        "vmin": 0,
    } 
    plot_metric(
        rmse_tensor,
        metric_name="RMSE",
        channel_names= variable_names,
        units= variable_units,
        save_path = save_path + "rmse.png",
        longitude = longitude,
        latitude = latitude,
        baselines=[RMSE(generated=baseline_unet, reference=reference_fine, pixel_wise=True)],
        baseline_names=["UNet-baseline"],
        **kwargs_rmse
    )
    print("plotting mae")
    kwargs_mae = {
        "vmin": 0,
        "cmap": "Oranges"
    } 
    plot_metric(
        mae_tensor,
        metric_name="MAE",
        channel_names= variable_names,
        units= variable_units,
        save_path = save_path + "mae.png",
        longitude = longitude,
        latitude = latitude,
        baselines=[MAE(generated=baseline_unet, reference=reference_fine, pixel_wise=True)],
        baseline_names=["UNet-baseline"],
        **kwargs_mae
    )
    print("plotting bias")
    bias_tensor_unet_baseline = bias_(generated=baseline_unet, reference=reference_fine, pixel_wise=True)
    for channel in range(bias_tensor.shape[0]) :
        bias = bias_tensor[channel]
        bias_unet_baseline = bias_tensor_unet_baseline[channel]
        vmin = min(torch.min(bias) , torch.min(bias_unet_baseline))
        vmax = max(torch.max(bias) , torch.max(bias_unet_baseline))
        vmax = max(abs(vmin),abs(vmax))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax = vmax)
        kwargs_bias = {
            "norm": norm,
            "cmap": "seismic"
        } 
        plot_metric(
            bias[None,...],
            metric_name="bias",
            channel_names= [variable_names[channel]],
            units= [variable_units[channel]],
            save_path = save_path + f"bias_{variable_names[channel]}.png",
            longitude = longitude,
            latitude = latitude,
            baselines=[bias_unet_baseline[None,...]],
            baseline_names=["UNet-baseline"],
            **kwargs_bias
            )
    print("plotting zero proportion")

    # Plot dry frequency map :  
    IPSL_AID.diagnostics.plot_dry_frequency_map(
        predictions = generated_all_members[:,5,:,:],  # Model predictions precipitation (fine predicted)
        targets = reference_fine[:,5,:,:],  # Ground truth precipitation (fine true)
        threshold = 0.1,  # threshold to define dry and wet (in mm)
        lat_1d = latitude,
        lon_1d=  longitude,
        filename="validation_dry_frequency_map.png",
        save_dir=save_path,
        figsize_multiplier=5,  # Base size per subplot
    )

    # Plot RAPSD :
    print("about to plot RAPSD")
    print(generated_all_members.shape)
    print(reference_fine.shape)
    print(reference_coarse.shape)
    plot_RAPSD(
        generated,
        reference_fine,
        reference_coarse = None,
        baselines=[baseline_unet],
        baseline_names=["UNet baseline"],
        norm=False,
        channel_rain=None,
        variables = variable_names,
        threshold_rain=None,
        save_path = save_path + "rapsd_no_norm.png",
        figsize_multiplier=4
    )

    # TODO uncomment when the inference has been done on run-free-2
    # IPSL_AID.diagnostics.plot_power_spectra(
    #     predictions = generated[0],
    #     targets = reference_fine,
    #     dlat = 0.25,
    #     dlon = 0.25,
    #     coarse_inputs = reference_coarse,
    #     variable_names = variable_names,
    #     save_dir = save_path,
    #     filename = "RAPSD.png"
    # )


    # Plot reliability diagrams :
    if nb_members > 1 :
        IPSL_AID.diagnostics.plot_ranks(
            predictions = generated,  # model predictions
            targets = reference_fine,  # ground truth
            variable_names= variable_names,  # list of variable names
            filename="ranks.png",
            save_dir=save_path,
            figsize_multiplier=4,
        )
    #Plot Spread Skill Ratio Hexbin :
    if nb_members > 1 :
        IPSL_AID.diagnostics.plot_spread_skill_ratio_hexbin(
            predictions = generated,
            targets = reference_fine,  # Ground truth (fine true)
            variable_names=variable_names,
            filename="spread_skill_ratio_hexbin.png",
            save_dir=save_path,
        )

    # Plot PDFs :
    plot_validation_pdfs(
        predictions = generated_all_members,  # Model predictions (fine predicted)
        targets = reference_fine,  # Ground truth (fine true)
        coarse_inputs=None,  # Coarse inputs for comparison (optional)
        baselines=[baseline_unet[0]],  # Baseline inputs for comparison (optional)
        baseline_names=["UNet"],  # Baseline names (optional)
        units=variable_units,
        variable_names=variable_names,  # List of variable names
        filename="validation_pdfs.png",
        save_dir=save_path,
        log_scale=[False, False, False, False, True, True],
        low_lim_percentile = 0.1, # percentile to use for lower limit of x axis
        high_lim_percentile = 99.9, # percentile to use for higher limit of x axis
        figsize_multiplier=5,  # Base size per subplot
        save_npz=False,
    )

    # plot PDFs :
    log_scale = [False, False, False, False, True, True]
    for i,var in enumerate(variable_names) :
        if var == "cape" or var == "precip" :
            low_lim = 0.1
            high_lim = 99.999
        else :
            low_lim = 0.1
            high_lim = 99.9
        plot_validation_pdfs(
            predictions = generated_all_members[:,i:i+1],  # Model predictions (fine predicted)
            targets = reference_fine[:,i:i+1],  # Ground truth (fine true)
            coarse_inputs=None,  # Coarse inputs for comparison (optional)
            baselines=[baseline_unet[0,:,i:i+1,:,:]],  # Baseline inputs for comparison (optional)
            baseline_names=["UNet"],  # Baseline names (optional)
            variable_names=[var],  # List of variable names
            units=[variable_units[i]],
            log_scale=log_scale[i],
            low_lim_percentile = low_lim, # percentile to use for lower limit of x axis
            high_lim_percentile = high_lim, # percentile to use for higher limit of x axis
            filename=f"validation_pdf_{var}.png",
            save_dir=save_path,
            figsize_multiplier=4,  # Base size per subplot
            save_npz=False,   
        )
        print(f"plotted pdf for {var}")

    # QQ-plots :
    for i,var in enumerate(variable_names) :
        if var == "CAPE" or var == "precip" :
            IPSL_AID.diagnostics.plot_qq_quantiles(
                predictions = generated_all_members[:,i:i+1,:,:],
                targets = reference_fine[:,i:i+1,:,:],
                coarse_inputs = reference_fine[:,i:i+1,:,:],
                variable_names=[var],
                units=[variable_units[i]],
                save_dir = save_path,
                filename =  f"low_qq_plot_{var}.png",
                quantiles = [0.01,0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            )
            IPSL_AID.diagnostics.plot_qq_quantiles(
                predictions = generated_all_members[:,i:i+1,:,:],
                targets = reference_fine[:,i:i+1,:,:],
                coarse_inputs = reference_fine[:,i:i+1,:,:],
                variable_names=[var],
                units=[variable_units[i]],
                save_dir = save_path,
                filename =  f"high_qq_plot_{var}.png",
                quantiles = [0.9,0.95,0.99,0.995,0.999,0.9995,0.9999]
            )
            print(f"plotted qq plot {var}")
        else :
            IPSL_AID.diagnostics.plot_qq_quantiles(
                predictions=generated_all_members[:,i:i+1,:,:],
                targets=reference_fine[:,i:i+1,:,:],
                coarse_inputs= reference_coarse[:,i:i+1,:,:],
                quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
                variable_names=[var],
                units=[variable_units[i]],
                save_dir = save_path,
                filename =  f"qq_plot_{var}.png"
            )

        print(f"plotted qq plot {var}")
        

    # Get divergence :
    div_gen = IPSL_AID.diagnostics.get_divergence( # TODO verify order of axes in gradient func
        u_tensor = generated_all_members[:,0,:,:],
        v_tensor = generated_all_members[:,1,:,:],
        spacing = 0.25,
    )
    div_ref =  IPSL_AID.diagnostics.get_divergence( # TODO verify order of axes in gradient func
        u_tensor = reference_fine[:,0,:,:],
        v_tensor = reference_fine[:,1,:,:],
        spacing = 0.25,
    )
    div_coarse =  IPSL_AID.diagnostics.get_divergence( # TODO verify order of axes in gradient func
        u_tensor = reference_coarse[:,0,:,:],
        v_tensor = reference_coarse[:,1,:,:],
        spacing = 0.25,
    )
    div_baseline_unet =  IPSL_AID.diagnostics.get_divergence( # TODO verify order of axes in gradient func
        u_tensor = baseline_unet[0,:,0,:,:],
        v_tensor = baseline_unet[0,:,1,:,:],
        spacing = 0.25,
    )
    plot_validation_pdfs(
        predictions = div_gen[:,None,:,:],  # Model predictions (fine predicted)
        targets = div_ref[:,None,:,:],  # Ground truth (fine true)
        coarse_inputs=div_coarse[:,None,:,:],  # Coarse inputs for comparison (optional)
        baselines=[div_baseline_unet[:,None,:,:]],  # Baseline inputs for comparison (optional)
        baseline_names=["UNet"],  # Baseline names (optional)
        variable_names=["divergence"],  # List of variable names
        units = ['m/(s*deg)'], #list of variable units
        filename="validation_pdfs_divergence.png",
        save_dir=save_path,
        log_scale=True,
        low_lim_percentile = 0.1, # percentile to use for lower limit of x axis
        high_lim_percentile = 99.9, # percentile to use for higher limit of x axis
        figsize_multiplier=4,  # Base size per subplot
        save_npz=False,
    )
    IPSL_AID.diagnostics.plot_qq_quantiles(
        predictions = div_gen[:,None,:,:],
        targets = div_ref[:,None,:,:],
        coarse_inputs = div_coarse[:,None,:,:],
        variable_names=["divergence"],
        units=["m/(s*deg)"],
        save_dir = save_path,
        filename =  "qq_plot_divergence.png",
        quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )

    # Get curl :
    curl_gen = IPSL_AID.diagnostics.get_curl( # TODO verify order of axes in gradient func
        u_tensor = generated_all_members[:,0,:,:],
        v_tensor = generated_all_members[:,1,:,:],
        spacing = 0.25,
    )
    curl_ref =  IPSL_AID.diagnostics.get_curl( # TODO verify order of axes in gradient func
        u_tensor = reference_fine[:,0,:,:],
        v_tensor = reference_fine[:,1,:,:],
        spacing = 0.25,
    )
    curl_coarse =  IPSL_AID.diagnostics.get_curl( # TODO verify order of axes in gradient func
        u_tensor = reference_coarse[:,0,:,:],
        v_tensor = reference_coarse[:,1,:,:],
        spacing = 0.25,
    )
    curl_baseline_unet =  IPSL_AID.diagnostics.get_curl( # TODO verify order of axes in gradient func
        u_tensor = baseline_unet[0,:,0,:,:],
        v_tensor = baseline_unet[0,:,1,:,:],
        spacing = 0.25,
    )
    plot_validation_pdfs(
        predictions = curl_gen[:,None,:,:],  # Model predictions (fine predicted)
        targets = curl_ref[:,None,:,:],  # Ground truth (fine true)
        coarse_inputs=curl_coarse[:,None,:,:],  # Coarse inputs for comparison (optional)
        baselines=[curl_baseline_unet[:,None,:,:]],  # Baseline inputs for comparison (optional)
        baseline_names=["UNet"],  # Baseline names (optional)
        variable_names=["curl"],  # List of variable names
        units = ['m/(s*deg)'], #list of variable units
        filename="validation_pdfs_curl.png",
        save_dir=save_path,
        log_scale=True,
        low_lim_percentile = 0.1, # percentile to use for lower limit of x axis
        high_lim_percentile = 99.9, # percentile to use for higher limit of x axis
        figsize_multiplier=4,  # Base size per subplot
        save_npz=False,
    )
    IPSL_AID.diagnostics.plot_qq_quantiles(
        predictions = curl_gen[:,None,:,:],
        targets = curl_ref[:,None,:,:],
        coarse_inputs = curl_coarse[:,None,:,:],
        variable_names=["curl"],
        units=["m/(s*deg)"],
        save_dir = save_path,
        filename =  "qq_plot_curl.png",
        quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )

    # get windspeed
    windspeed_gen = get_windspeed(u = generated_all_members[:,0,:,:], v = generated_all_members[:,1,:,:])
    windspeed_ref = get_windspeed(u = reference_fine[:,0,:,:], v = reference_fine[:,1,:,:])
    windspeed_coarse = get_windspeed(u = reference_coarse[:,0,:,:], v = reference_coarse[:,1,:,:])
    windspeed_baseline_unet = get_windspeed(u = baseline_unet[0,:,0,:,:], v = baseline_unet[0,:,1,:,:])

    plot_validation_pdfs( 
        predictions=windspeed_gen[:,None,:,:],
        targets=windspeed_ref[:,None,:,:],
        variable_names=["wind speed 850"],
        coarse_inputs=windspeed_coarse[:,None,:,:],
        baselines=[windspeed_baseline_unet[:,None,:,:]],
        baseline_names=["UNet"],
        units=['m/s'],
        low_lim_percentile = 0.1, # percentile to use for lower limit of x axis
        high_lim_percentile = 99.9, # percentile to use for higher limit of x axis
        log_scale=False,
        save_dir = save_path,
        filename="pdf_windspeed.png",
    )

    IPSL_AID.diagnostics.plot_qq_quantiles(
        predictions = windspeed_gen[:,None,:,:],
        targets = windspeed_ref[:,None,:,:],
        coarse_inputs = windspeed_coarse[:,None,:,:],
        variable_names=["windspeed 850 hPa"],
        units=["m/s"],
        save_dir = save_path,
        filename =  "qq_plot_windspeed.png",
        quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    )

    #TODO change to custom RAPSD func
    # windspeed_gen = get_windspeed(u = generated[0,:,0,:,:], v = generated[0,:,1,:,:])
    # IPSL_AID.diagnostics.plot_power_spectra(
    #     predictions = windspeed_gen[:,None,:,:],  # Model predictions
    #     targets = windspeed_ref[:,None,:,:],  # Ground truth
    #     dlat = 0.25,  # Grid spacing in latitude (degrees)
    #     dlon = 0.25,  # Grid spacing in longitude (degrees)
    #     coarse_inputs=  windspeed_coarse[:,None,:,:],  # Coarse inputs for comparison (optional)
    #     variable_names= ["windspeed 850 hPa"],  # List of variable names
    #     filename="power_spectra_windspeed.png",
    #     save_dir=save_path,
    #     figsize_multiplier=4,
    #     save_npz=False,
    # )
    plot_RAPSD(
        windspeed_gen[None,:,None,:,:],
        windspeed_ref[:,None,:,:],
        reference_coarse = windspeed_coarse[:,None,:,:],
        baselines=[windspeed_baseline_unet[:,None,:,:]],
        baseline_names=["UNet baseline"],
        norm=False,
        channel_rain=None,
        variables = ["Windspeed 850 hPa"],
        threshold_rain=None,
        save_path = save_path + "rapsd_windspeed_no_norm.png",
        figsize_multiplier=4
    )

    nb_extremes_above_threshold(
        generated = generated[:,:,5,:,:],
        coarse = None,
        fine = reference_fine[:,5,:,:],
        baselines=[baseline_unet[:,:,5,:,:]],
        baseline_names=["UNet baseline"],
        lon = longitude,
        lat = latitude,
        variable_name="precip",
        unit="mm/6h",
        threshold = 15,
        save_path = save_path + "freq_over_extremes_15mm.png"
    )
    nb_extremes_above_threshold(
        generated = generated[:,:,5,:,:],
        coarse = None,
        fine = reference_fine[:,5,:,:],
        baselines=[baseline_unet[:,:,5,:,:]],
        baseline_names=["UNet baseline"],
        lon = longitude,
        lat = latitude,
        variable_name="precip",
        unit="mm/6h",
        threshold = 25,
        save_path = save_path + "freq_over_extremes_25mm.png"
    )
    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_folder")
    parser.add_argument("-pattern_data_files")
    parser.add_argument("-nb_members")
    parser.add_argument("-path_log")
    parser.add_argument("-save_path")
    parser.add_argument(
        "--variable_names",
        type=str,
        nargs="+",
        default=["u850", "v850", "t850", "TCWV"]
    )
    parser.add_argument(
        "--variable_units",
        type=str,
        nargs="+",
        default=["m/s", "m/s", "K", "mm"]
    )
    parser.add_argument(
        "--diagnostic_names",
        type=str,
        nargs="+",
        default=None
    )
    parser.add_argument(
        "--diagnostic_units",
        type=str,
        nargs="+",
        default=None
    )    
    parser.add_argument('--path_folder_Unet', nargs='?', default="/lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDZ-ERA/unet_baseline_04_06_2026/results/", help='path to the Unet generated data for baseline.')
    parser.add_argument('--path_folder_bilinear', nargs='?', default=None, help='path to the bilinear interpolation data for baseline.')
    parser.add_argument('--path_folder_aux', nargs="?", default=None, help="path to any other generated data for baseline.")
    parser.add_argument(
        "--convert_to_mm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="multiply precip by 1000 to convert from m to mm",
    )
    parser.add_argument(
        "--convert_baseline_to_mm",
        type=lambda x: x.lower() == "true",
        default=True,
        help="multiply precip by 1000 to convert from m to mm for baseline",
    )
    args = parser.parse_args()
    print("imports successful")
    main_full_vars(
        path_folder = args.path_folder,
        pattern_data_files = args.pattern_data_files,
        nb_members = int(args.nb_members),
        path_log = args.path_log,
        save_path = args.save_path,
        variable_names = args.variable_names,
        variable_units = args.variable_units,
        path_folder_Unet = args.path_folder_Unet,
        convert_to_mm = args.convert_to_mm,
        convert_baseline_to_mm = args.convert_baseline_to_mm,
    )





























    