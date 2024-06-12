set -ex

export PYTHONPATH="."

# Single step predictor
export SS_ARGS="--horizon 2 --out_obs_horizon 1 --in_obs_horizon 1 --in_act_horizon 1 --model_type state_predictor --ensemble_size 5"
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_250_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
    --save_path feeding_250_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type Feeding \
    $SS_ARGS
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_100_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_100.zarr/ \
    --save_path feeding_100_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type Feeding \
    $SS_ARGS
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_50_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_50.zarr/ \
    --save_path feeding_50_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type Feeding \
    $SS_ARGS
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/drinking_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/Drinking_1000k_ppo.zarr/ \
    --save_path drinking_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type Drinking \
    $SS_ARGS --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/arm_manipulation_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ArmManipulation_ppo.zarr/ \
    --save_path arm_manipulation_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type ArmManipulation \
    $SS_ARGS --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/scratch_itch_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ScratchItch_ppo.zarr/ \
    --save_path scratch_itch_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type ScratchItch \
    $SS_ARGS --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/bed_bathing_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/BedBathing_ppo.zarr/ \
    --save_path bed_bathing_ensemble_state_prediction_sm_To1_Ta1_H2_O1.pth \
    --env_type BedBathing \
    $SS_ARGS --num_epochs 10
