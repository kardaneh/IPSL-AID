#!/bin/bash

#SBATCH --job-name=inference_unet_corrdiff
#SBATCH -A huw@v100 #selectionner l'account
##SBATCH -C v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10   #nombre de CPU par tache pour gpu_p5 (8 pour a100, 1/8 des CPU du noeud 8-GPU A100)
#SBATCH --output=./log/inference_unet_corrdiff_test%j.log
#SBATCH --error=./log/inference_unet_corrdiff_error%j.log
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:30:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-dev #qos pour test à enlever dans les gros entrainements

module purge
module load pytorch-gpu/py3/2.6.0

python -u ../main_lmdz_era.py --run_type inference \
--model_name unet \
--arch adm \
--precond unet \
--custom_loss false \
--num_members 1 \
--save_model True \
--inference_type direct \
--num_steps 20 \
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
--units_list m/s m/s K kg/kg --num_epochs 50 --num_workers 8 \
--datadir /lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/ \
--datadir_coarse /lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/ \
--per_var_datadir_train var131=u850_europe_6h_1985_1999_sorted.zarr var132=v850_europe_6h_1985_1999_sorted.zarr var130=t850_europe_6h_1985_1999_sorted.zarr var137=TCWV_europe_6h_1985_1999_sorted.zarr \
--per_var_datadir_valid var131=u850_europe_6h_1985_1999_sorted.zarr var132=v850_europe_6h_1985_1999_sorted.zarr var130=t850_europe_6h_1985_1999_sorted.zarr var137=TCWV_europe_6h_1985_1999_sorted.zarr \
--per_coarse_var_datadir_train u850=u850_france_6h_1985_1999_sorted_regrid_bilinear.zarr v850=v850_france_6h_1985_1999_sorted_regrid_bilinear.zarr t850=t850_france_6h_1985_1999_sorted_regrid_bilinear.zarr prw=prw_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_coarse_var_datadir_valid u850=u850_france_6h_1985_1999_sorted_regrid_bilinear.zarr v850=v850_france_6h_1985_1999_sorted_regrid_bilinear.zarr t850=t850_france_6h_1985_1999_sorted_regrid_bilinear.zarr prw=prw_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_dynamic_cov_datadir_train u700=u700_france_6h_1985_1999_sorted_regrid_bilinear.zarr u500=u500_france_6h_1985_1999_sorted_regrid_bilinear.zarr v700=v700_france_6h_1985_1999_sorted_regrid_bilinear.zarr v500=v500_france_6h_1985_1999_sorted_regrid_bilinear.zarr t700=t700_france_6h_1985_1999_sorted_regrid_bilinear.zarr t500=t500_france_6h_1985_1999_sorted_regrid_bilinear.zarr q850=q850_france_6h_1985_1999_sorted_regrid_bilinear.zarr q700=q700_france_6h_1985_1999_sorted_regrid_bilinear.zarr q500=q500_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_dynamic_cov_datadir_valid u700=u700_france_6h_1985_1999_sorted_regrid_bilinear.zarr u500=u500_france_6h_1985_1999_sorted_regrid_bilinear.zarr v700=v700_france_6h_1985_1999_sorted_regrid_bilinear.zarr v500=v500_france_6h_1985_1999_sorted_regrid_bilinear.zarr t700=t700_france_6h_1985_1999_sorted_regrid_bilinear.zarr t500=t500_france_6h_1985_1999_sorted_regrid_bilinear.zarr q850=q850_france_6h_1985_1999_sorted_regrid_bilinear.zarr q700=q700_france_6h_1985_1999_sorted_regrid_bilinear.zarr q500=q500_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_diagnostic_var_datadir_train var59=cape_europe_6h_1985_1999_sorted.zarr var228=precip_europe_6h_1985_1999_sorted.zarr \
--per_diagnostic_var_datadir_valid var59=cape_europe_6h_1985_1999_sorted.zarr var228=precip_europe_6h_1985_1999_sorted.zarr \
--main_folder /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA \
--sub_folder unet_baseline_05_10_2026_trained_on_1985-1999_inference_1985-1999 \
--in_channels 15 --cond_channels 0 --out_channels 6 \
--load_checkpoint_name /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA/unet_baseline_05_10_2026_trained_on_1985-1999/checkpoints/run_best_model.pth.tar \
--batch_size 32 \
--multiplication_factor_precip_coarse 1000 \
--multiplication_factor_precip_fine 1000