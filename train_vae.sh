set -ex

export PYTHONPATH="."

# VAE Predictor
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_250_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
    --save_path ensemble_vae_sm_250_To8_Ta16_E5.pth \
    --env_type Feeding \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_100_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_100.zarr/ \
    --save_path ensemble_vae_sm_100_To8_Ta16_E5.pth \
    --env_type Feeding \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_50_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_50.zarr/ \
    --save_path ensemble_vae_sm_50_To8_Ta16_E5.pth \
    --env_type Feeding \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/drinking_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/Drinking_1000k_ppo.zarr/ \
    --save_path drinking_ensemble_vae_sm_250_To8_Ta16_E5.pth \
    --env_type Drinking \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/arm_manipulation_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ArmManipulation_ppo.zarr/ \
    --save_path arm_manipulation_ensemble_vae_sm_250_To8_Ta16_E5.pth \
    --env_type ArmManipulation \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/scratch_itch_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ScratchItch_ppo.zarr/ \
    --save_path scratch_itch_ensemble_vae_sm_250_To4_Ta8_E5.pth \
    --env_type ScratchItch \
    --horizon 9 --out_obs_horizon 1 --in_obs_horizon 4 --in_act_horizon 8 --model_type vae --ensemble_size 5
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/bed_bathing_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/BedBathing_ppo.zarr/ \
    --save_path bed_bathing_ensemble_vae_sm_250_To8_Ta16_E5.pth \
    --env_type BedBathing \
    --horizon 17 --out_obs_horizon 1 --in_obs_horizon 8 --in_act_horizon 16 --model_type vae --ensemble_size 5
