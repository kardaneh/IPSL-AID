#!/bin/bash
#SBATCH --job-name=diffusion_prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct_bs36_lr0d0001_20260513_120904
#SBATCH --account=EUHPC_D27_095
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=64G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --error=/leonardo_work/EUHPC_D27_095/IPSL-AID/slurm_io/diffusion_prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct_bs36_lr0d0001_20260513_120904_%j.err
#SBATCH --output=/leonardo_work/EUHPC_D27_095/IPSL-AID/slurm_io/diffusion_prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct_bs36_lr0d0001_20260513_120904_%j.out
# #SBATCH --mail-user=kkingston@ipsl.fr
# #SBATCH --mail-type=END,FAIL

ulimit -s unlimited
module purge

# Activate virtual environment
source .venv/bin/activate

export PYTHONUNBUFFERED=1

echo "===== Job Infos ====="
echo "Debug mode: false"
echo "Apply filter: false"
echo "Inference type: direct"
echo "Node list: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Current dir: $(pwd)"
echo "Output main folder: prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter"
echo "Output sub folder: train_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct_bs36_lr0d0001"
echo "====================="

# Use ipsl-aid command directly
ipsl-aid \
    --debug false \
    --main_folder "prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter" \
    --sub_folder "train_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct_bs36_lr0d0001" \
    --prefix "run_20260513_120904" \
    --run_type "train" \
    --save_model true \
    --save_checkpoint_name "run_20260513_120904_final_model_epoch14.pth.tar" \
    --load_checkpoint_name "run_20260513_120904_final_model_epoch14.pth.tar" \
    --save_per_samples "10000" \
    --inference_type "direct" \
    --arch "ddpmpp" \
    --precond "unet" \
    --in_channels "10" \
    --out_channels "6" \
    --apply_filter false \
    --model_name "prod_y2015_2019_norm_cos_sin_lat145_lon361_vars6_dtfp32_archddpmpp_preunet_in10_out6_nofilter_ep10_sb12_tb1800_eps0d02_mrg8_inf_direct" \
    --varnames_list VAR_2T VAR_10U VAR_10V VAR_TP VAR_D2M VAR_ST \
    --normalization_types VAR_2T=standard VAR_10U=standard VAR_10V=standard VAR_TP=log1p_standard VAR_D2M=standard VAR_ST=standard \
    --constant_varnames_list z lsm \
    --constant_varnames_file "ERA5_const_sfc_variables.nc" \
    --units_list K m/s m/s m/h K K \
    --year_start "2015" \
    --year_end "2019" \
    --year_start_test "2020" \
    --year_end_test "2020" \
    --batch_size "36" \
    --num_epochs "21" \
    --learning_rate "0.0001" \
    --num_workers "16" \
    --datadir "/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily" \
    --per_var_datadir VAR_2T=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily VAR_10U=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily VAR_10V=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily VAR_TP=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily_tp VAR_D2M=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily_d2m_sstk VAR_ST=/leonardo_work/EUHPC_D27_095/kkingston/AI-Downscaling/data/data_FOURxDaily_st \
    --time_normalization "cos_sin" \
    --tbatch "1800" \
    --sbatch "12" \
    --batch_size_lat "145" \
    --batch_size_lon "361" \
    --epsilon "0.02" \
    --beta "1.0" \
    --margin "8" \
    --dynamic_covariates_dir "../data_covariates/" \
    --dtype "fp32" \
    --num_steps "10" \
    --sigma_min "0.002" \
    --sigma_max "80.0" \
    --rho "7" \
    --s_churn "40" \
    --solver "heun" \
    --compute_crps "false"

exit $?
