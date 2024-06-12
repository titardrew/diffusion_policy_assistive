set -ex

export PYTHONPATH="."

# Multi step predictor 
export MS_ARGS="--horizon 12 --out_obs_horizon 3 --in_obs_horizon 2 --in_act_horizon 3 --model_type state_predictor --ensemble_size 5"
python safety_model/train.py \
    --test_parquet_path parquet_datasets/feeding_250_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
    --save_path feeding_250_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type Feeding \
    --tb_experiment_path runs/feeding_250_ensemble_state_prediction_sm_To3_Ta3_H8_O3 \
    $MS_ARGS
exit()
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_100_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_100.zarr/ \
    --save_path feeding_100_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type Feeding \
    $MS_ARGS
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/feeding_50_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1_50.zarr/ \
    --save_path feeding_50_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type Feeding \
    $MS_ARGS
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/drinking_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/Drinking_1000k_ppo.zarr/ \
    --save_path drinking_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type Drinking \
    $MS_ARGS    # --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/arm_manipulation_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ArmManipulation_ppo.zarr/ \
    --save_path arm_manipulation_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type ArmManipulation \
    $MS_ARGS    # --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/scratch_itch_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/ScratchItch_ppo.zarr/ \
    --save_path scratch_itch_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type ScratchItch \
    $MS_ARGS    # --num_epochs 10
python diffusion_policy/scripts/assistive_train_safety_model.py \
    --test_parquet_path parquet_datasets/bed_bathing_test.parquet \
    --zarr_path /home/aty/Data/AssistiveDiffusion/Datasets/BedBathing_ppo.zarr/ \
    --save_path bed_bathing_ensemble_state_prediction_sm_To3_Ta3_H8_O3.pth \
    --env_type BedBathing \
    $MS_ARGS    # --num_epochs 10
