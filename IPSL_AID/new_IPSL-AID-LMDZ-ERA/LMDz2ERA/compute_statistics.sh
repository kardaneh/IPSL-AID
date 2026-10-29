#!/bin/bash

#SBATCH --job-name=unet_era5coarse2era5
#SBATCH -A huw@cpu #selectionner l'account
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20   #nombre de CPU par tache pour gpu_p5 (8 pour a100, 1/8 des CPU du noeud 8-GPU A100)
#SBATCH --output=./log/compute_stats_output%j.log
#SBATCH --error=./log/compute_stats_error%j.log
##SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=01:00:00              # temps maximum d'execution demande (HH:MM:SS)

# dont load any venv before launching job
# make sure that zarr version installed is 3.1.4
module purge
module load pytorch-gpu/py3/2.6.0

python -u ./compute_statistics.py --run_type train \
--model_name unet \
--arch adm \
--precond unet \
--custom_loss false \
--num_members 1 \
--save_model True \
--inference_type direct \
--domain_lon -6 9.75 \
--domain_lat 54.75 39 \
--varnames_list var131 var132 var130 var137 \
--coarse_varnames_list u850 v850 t850 prw \
--diagnostics_varnames_list var59 var228 \
--constant_varnames_list var172 var129 \
--constant_varnames_file static_cov.zarr \
--normalization_types var131=standard var132=standard var130=standard var137=standard \
--normalization_types_coarse u850=standard v850=standard t850=standard prw=standard \
--normalization_types_diagnostics var59=log var228=log \
--normalization_types_dynamic_cov u700=standard u500=standard v700=standard v500=standard t700=standard t500=standard q850=standard q700=standard q500=standard \
--path_folder_stats ./stats/ \
--json_stats_rawdata statistics_lmdz_nudg_coarse_1985-1999.json \
--json_stats_residual statistics_lmdz_era5_residual_1985-1999.json \
--json_stats_diag statistics_lmdz_nudg_coarse_1985-1999.json \
--dynamic_covariates u700 u500 v700 v500 t700 t500 q850 q700 q500 \
--datadir_dynamic_cov /lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/ \
--datadir_diagnostics /lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/ \
--units_list m/s m/s K kg/kg --num_epochs 25 --num_workers 15 \
--datadir /lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/ \
--datadir_coarse /lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/ \
--per_var_datadir_train var131=u850_europe_6h_1985_1999_sorted.zarr var132=v850_europe_6h_1985_1999_sorted.zarr var130=t850_europe_6h_1985_1999_sorted.zarr var137=TCWV_europe_6h_1985_1999_sorted.zarr \
--per_var_datadir_valid var131=u850_europe_6h_2000_2014_sorted.zarr var132=v850_europe_6h_2000_2014_sorted.zarr var130=t850_europe_6h_2000_2014_sorted.zarr var137=TCWV_europe_6h_2000_2014_sorted.zarr \
--per_coarse_var_datadir_train u850=u850_france_6h_1985_1999_sorted_regrid_bilinear.zarr v850=v850_france_6h_1985_1999_sorted_regrid_bilinear.zarr t850=t850_france_6h_1985_1999_sorted_regrid_bilinear.zarr prw=prw_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_coarse_var_datadir_valid u850=u850_france_6h_2000_2014_sorted_regrid_bilinear.zarr v850=v850_france_6h_2000_2014_sorted_regrid_bilinear.zarr t850=t850_france_6h_2000_2014_sorted_regrid_bilinear.zarr prw=prw_france_6h_2000_2014_sorted_regrid_bilinear.zarr \
--per_dynamic_cov_datadir_train u700=u700_france_6h_1985_1999_sorted_regrid_bilinear.zarr u500=u500_france_6h_1985_1999_sorted_regrid_bilinear.zarr v700=v700_france_6h_1985_1999_sorted_regrid_bilinear.zarr v500=v500_france_6h_1985_1999_sorted_regrid_bilinear.zarr t700=t700_france_6h_1985_1999_sorted_regrid_bilinear.zarr t500=t500_france_6h_1985_1999_sorted_regrid_bilinear.zarr q850=q850_france_6h_1985_1999_sorted_regrid_bilinear.zarr q700=q700_france_6h_1985_1999_sorted_regrid_bilinear.zarr q500=q500_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_dynamic_cov_datadir_valid u700=u700_france_6h_2000_2014_sorted_regrid_bilinear.zarr u500=u500_france_6h_2000_2014_sorted_regrid_bilinear.zarr v700=v700_france_6h_2000_2014_sorted_regrid_bilinear.zarr v500=v500_france_6h_2000_2014_sorted_regrid_bilinear.zarr t700=t700_france_6h_2000_2014_sorted_regrid_bilinear.zarr t500=t500_france_6h_2000_2014_sorted_regrid_bilinear.zarr q850=q850_france_6h_2000_2014_sorted_regrid_bilinear.zarr q700=q700_france_6h_2000_2014_sorted_regrid_bilinear.zarr q500=q500_france_6h_2000_2014_sorted_regrid_bilinear.zarr \
--per_diagnostic_var_datadir_train var59=cape_europe_6h_1985_1999_sorted.zarr var228=precip_europe_6h_1985_1999_sorted.zarr \
--per_diagnostic_var_datadir_valid var59=cape_europe_6h_2000_2014_sorted.zarr var228=precip_europe_6h_2000_2014_sorted.zarr \
--main_folder /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA \
--sub_folder unet_baseline_03_10_2026 \
--in_channels 15 --cond_channels 0 --out_channels 6 \
--load_checkpoint_name /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-ERACOARSE2ERA/unet_baseline_22_09_2026/checkpoints/run_best_model.pth.tar \
--batch_size 32 \
--multiplication_factor_precip_coarse 1000 \
--multiplication_factor_precip_fine 1000

# 6 (bruit) + 4 (coarse vars) + 10 (dynamic cov) + 2 (static cov)