#!/bin/bash

#SBATCH --job-name=diff_era5coarse2era5
#SBATCH -A huw@v100 #selectionner l'account
##SBATCH -C v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10   #nombre de CPU par tache pour gpu_p5 (8 pour a100, 1/8 des CPU du noeud 8-GPU A100)
#SBATCH --output=./log/diff_infer_output%j.log
#SBATCH --error=./log/diff_infer_error%j.log
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=04:00:00              # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --qos=qos_gpu-dev #qos pour test à enlever dans les gros entrainements

# dont load any venv before launching job
# make sure that zarr version installed is 3.1.4
module purge
module load pytorch-gpu/py3/2.6.0

python -u ../main_lmdz_era.py --run_type inference \
--arch adm \
--precond edm \
--custom_loss false \
--num_members 1 \
--save_model True \
--inference_type sampler \
--num_steps 50 \
--domain_lon -6 9.75 \
--domain_lat 54.75 39 \
--varnames_list var131 var132 var130 var137 var59 var228 \
--coarse_varnames_list var131 var132 var130 var137 var59 var228 \
--constant_varnames_list var172 var129 \
--constant_varnames_file static_cov.zarr \
--normalization_types var131=standard var132=standard var130=standard var137=standard var59=log var228=log \
--normalization_types_coarse var131=standard var132=standard var130=standard var137=standard var59=log var228=log \
--normalization_types_dynamic_cov u700=standard u500=standard v700=standard v500=standard t700=standard t500=standard q850=standard q700=standard q500=standard \
--path_folder_stats ./stats_corrdiff/ \
--json_stats_rawdata statistics_unet_coarse_1985-1999.json \
--json_stats_residual statistics_unet_coarse_era5_residual_1985-1999.json \
--json_stats_diag statistics_unet_coarse_1985-1999.json \
--dynamic_covariates u700 u500 v700 v500 t700 t500 q850 q700 q500 \
--datadir_dynamic_cov /lustre/fsn1/projects/rech/huw/uzb76ep/LMDZ/run-nudg-6h-1/bilinear/ \
--units_list m/s m/s K kg/kg J/kg mm/6h --num_epochs 50 --num_workers 8 \
--datadir /lustre/fsn1/projects/rech/huw/uzb76ep/ERA5_Europe/ \
--per_var_datadir_train var131=u850_europe_6h_1985_1999_sorted.zarr var132=v850_europe_6h_1985_1999_sorted.zarr var130=t850_europe_6h_1985_1999_sorted.zarr var137=TCWV_europe_6h_1985_1999_sorted.zarr var59=cape_europe_6h_1985_1999_sorted.zarr var228=precip_europe_6h_1985_1999_sorted.zarr \
--per_var_datadir_valid var131=u850_europe_6h_2000_2014_sorted.zarr var132=v850_europe_6h_2000_2014_sorted.zarr var130=t850_europe_6h_2000_2014_sorted.zarr var137=TCWV_europe_6h_2000_2014_sorted.zarr var59=cape_europe_6h_2000_2014_sorted.zarr var228=precip_europe_6h_2000_2014_sorted.zarr \
--per_coarse_var_datadir_train var131=predictions_unet_corrdiff_1985_1999.zarr var132=predictions_unet_corrdiff_1985_1999.zarr var130=predictions_unet_corrdiff_1985_1999.zarr var137=predictions_unet_corrdiff_1985_1999.zarr var59=predictions_unet_corrdiff_1985_1999.zarr var228=predictions_unet_corrdiff_1985_1999.zarr \
--per_coarse_var_datadir_valid var131=predictions_unet_corrdiff_2000_2014.zarr var132=predictions_unet_corrdiff_2000_2014.zarr var130=predictions_unet_corrdiff_2000_2014.zarr var137=predictions_unet_corrdiff_2000_2014.zarr var59=predictions_unet_corrdiff_2000_2014.zarr var228=predictions_unet_corrdiff_2000_2014.zarr \
--datadir_coarse /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA/unet_baseline_05_10_2026_trained_on_1985-1999_inference_1985-1999/results/ \
--per_dynamic_cov_datadir_train u700=u700_france_6h_1985_1999_sorted_regrid_bilinear.zarr u500=u500_france_6h_1985_1999_sorted_regrid_bilinear.zarr v700=v700_france_6h_1985_1999_sorted_regrid_bilinear.zarr v500=v500_france_6h_1985_1999_sorted_regrid_bilinear.zarr t700=t700_france_6h_1985_1999_sorted_regrid_bilinear.zarr t500=t500_france_6h_1985_1999_sorted_regrid_bilinear.zarr q850=q850_france_6h_1985_1999_sorted_regrid_bilinear.zarr q700=q700_france_6h_1985_1999_sorted_regrid_bilinear.zarr q500=q500_france_6h_1985_1999_sorted_regrid_bilinear.zarr \
--per_dynamic_cov_datadir_valid u700=u700_france_6h_2000_2014_sorted_regrid_bilinear.zarr u500=u500_france_6h_2000_2014_sorted_regrid_bilinear.zarr v700=v700_france_6h_2000_2014_sorted_regrid_bilinear.zarr v500=v500_france_6h_2000_2014_sorted_regrid_bilinear.zarr t700=t700_france_6h_2000_2014_sorted_regrid_bilinear.zarr t500=t500_france_6h_2000_2014_sorted_regrid_bilinear.zarr q850=q850_france_6h_2000_2014_sorted_regrid_bilinear.zarr q700=q700_france_6h_2000_2014_sorted_regrid_bilinear.zarr q500=q500_france_6h_2000_2014_sorted_regrid_bilinear.zarr \
--main_folder /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA \
--sub_folder diffusion_corrdiff_06_10_2026_trained_on_1985-1999_inference_2000-2014 \
--in_channels 6 --cond_channels 17 --out_channels 6 \
--load_checkpoint_name /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA/diffusion_corrdiff_06_10_2026_trained_on_1985-1999/checkpoints/run_epoch0049_diffusion_model_checkpoint.pth.tar \
--batch_size 32 \
--multiplication_factor_precip_coarse 1 \
--multiplication_factor_precip_fine 1000

# 6 (bruit) + 6 (coarse vars) + 9 (dynamic cov) + 2 (static cov)