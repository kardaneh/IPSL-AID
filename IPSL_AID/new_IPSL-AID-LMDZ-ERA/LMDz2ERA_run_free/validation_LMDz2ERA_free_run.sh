#!/bin/bash

#SBATCH --job-name=validation_ERACOARSE2ERA
#SBATCH -A huw@cpu #selectionner l'account
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20   #nombre de CPU par tache pour gpu_p5 (8 pour a100, 1/8 des CPU du noeud 8-GPU A100)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=01:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=./log/validation_eracoarse2era_output%j.log
#SBATCH --error=./log/validation_eracoarse2era_error%j.log

# we ask for extra memory because we load large files into memory

module purge
module load pytorch-gpu/py3/2.6.0

cd /lustre/fswork/projects/rech/huw/uzb76ep/Downscaling_JZ/new_IPSL-AID-LMDZ-ERA/LMDz2ERA_run_free/results/
mkdir diffusion_corrdiff_06_10_2026_trained_on_1985-1999_inference_2000-2014_run_free

srun python -u /lustre/fswork/projects/rech/huw/uzb76ep/Downscaling_JZ/new_IPSL-AID-LMDZ-ERA/validation.py \
-path_folder /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA_run_free/diffusion_corrdiff_06_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/results/ \
-pattern_data_files test_save_validation \
-nb_members 1 \
-path_log /lustre/fswork/projects/rech/huw/uzb76ep/Downscaling_JZ/new_IPSL-AID-LMDZ-ERA/LMDz2ERA_run_free/results/diffusion_corrdiff_06_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/ \
-save_path /lustre/fswork/projects/rech/huw/uzb76ep/Downscaling_JZ/new_IPSL-AID-LMDZ-ERA/LMDz2ERA_run_free/results/diffusion_corrdiff_06_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/ \
--path_folder_Unet /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA_run_free/unet_baseline_05_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/results/ \
--variable_names u850 v850 t850 TCWV CAPE precip \
--variable_units m/s m/s K mm J/kg mm/6h \
--convert_to_mm false \
--convert_baseline_to_mm false \