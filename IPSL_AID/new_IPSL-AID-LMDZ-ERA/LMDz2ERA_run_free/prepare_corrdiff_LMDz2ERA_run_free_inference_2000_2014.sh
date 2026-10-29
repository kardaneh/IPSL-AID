#!/bin/bash

#SBATCH --job-name=unet_era5coarse2era5
#SBATCH -A huw@cpu #selectionner l'account
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20   #nombre de CPU par tache pour gpu_p5 (8 pour a100, 1/8 des CPU du noeud 8-GPU A100)
##SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=00:10:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=./log/prepare_corrdiff_output%j.log
#SBATCH --error=./log/prepare_corrdiff_error%j.log

# dont load any venv before launching job
# make sure that zarr version installed is 3.1.4
module purge
module load pytorch-cpu/py3/1.7.1

python -u ./prepare_corrdiff_ERACoarse2ERA.py \
--dir_coarse /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA_run_free/unet_baseline_05_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/results/ \
--dir_coarse_test_set /lustre/fsn1/projects/rech/huw/uzb76ep/IPSL-AID-LMDz2ERA_run_free/unet_baseline_05_10_2026_trained_on_1985-1999_inference_2000-2014_run_free/results/ \
--period_train 1985_1999 \
--period_test 2000_2014

# 6 (bruit) + 4 (coarse vars) + 10 (dynamic cov) + 2 (static cov)