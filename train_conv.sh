set -ex

export PYTHONPATH="."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Multi step predictor 
# export MS_ARGS="--num_epochs 100 --out_obs_horizon 1 --in_horizon 5 --gap_horizon 3 --model_type state_predictor --ensemble_size 5"
# python safety_model/train.py \
#     --test_parquet_path parquet_datasets/feeding_250_test.parquet \
#     --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
#     --test_zarr_path out_voraus/feeding_250/zarr_recording/cat.zarr/ \
#     --save_path test.pth \
#     --env_type Feeding \
#     --metric nll \
#     --tb_experiment_path runs/test_incvar_conv_gap3_in5_out1_time_nll \
#     $MS_ARGS


# # VAE
# export MS_ARGS="--num_epochs 100 --in_horizon 16 --gap_horizon 0 --out_obs_horizon 0 --model_type vae"
# python safety_model/train.py \
#     --test_parquet_path parquet_datasets/feeding_250_test.parquet \
#     --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
#     --test_zarr_path out_voraus/feeding_250/zarr_recording/cat.zarr/ \
#     --save_path test2.pth \
#     --env_type Feeding \
#     --metric recon \
#     --batch_size 64 \
#     --ensemble_size 2 \
#     --tb_experiment_path runs/test_vae_incvar_full \
#     --device cuda \
#     $MS_ARGS

# CVAE
# Too long...
# export MS_ARGS="--num_epochs 1000 --full_episode --in_horizon 0 --gap_horizon 0 --out_obs_horizon 16 --model_type cvae"
# python safety_model/train.py \
#     --test_parquet_path parquet_datasets/feeding_250_test.parquet \
#     --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
#     --test_zarr_path out_voraus/feeding_250/zarr_recording/cat.zarr/ \
#     --save_path test_cvae.pth \
#     --env_type Feeding \
#     --metric recon \
#     --batch_size 64 \
#     --ensemble_size 1 \
#     --kl_weight 10 \
#     --test_freq 5 \
#     --tb_experiment_path runs/test_cvae_i0_g0_o16 \
#     --device cuda \
#     $MS_ARGS

export MS_ARGS="--num_epochs 1000 --in_horizon 5 --gap_horizon 3 --out_obs_horizon 1 --model_type cvae"
python safety_model/train.py \
    --test_parquet_path parquet_datasets/feeding_250_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
    --test_zarr_path out_voraus/feeding_250/zarr_recording/cat.zarr/ \
    --save_path test_cvae_i5_g3_o1.pth \
    --env_type Feeding \
    --metric recon \
    --batch_size 64 \
    --ensemble_size 1 \
    --kl_weight 10 \
    --test_freq 5 \
    --backend tb \
    --project test \
    --use_maximum \
    --experiment_path runs/test_cvae_i5_g3_o1_kl10 \
    --device cuda \
    $MS_ARGS
#    --lr 1e-4 \
#    --use_times \