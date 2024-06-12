#!/bin/bash
echo feeding 250
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt -o out/feeding_250_none
echo feeding 250 state predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt -o out/feeding_250_state_predictor --safety_model feeding_250_state_predictor
echo feeding 250 vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt -o out/feeding_250_vae --safety_model feeding_250_vae

echo feeding 100
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1-100/horizon_10_n_obs_2_n_act_2/checkoint/best.ckpt -o out/feeding_100_none
echo feeding 100 state predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1-100/horizon_10_n_obs_2_n_act_2/checkoint/best.ckpt -o out/feeding_100_state_predictor --safety_model feeding_100_state_predictor
echo feeding 100 vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1-100/horizon_10_n_obs_2_n_act_2/checkoint/best.ckpt -o out/feeding_100_vae --safety_model feeding_100_vae

echo drinking
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/DrinkingJaco_1kk/latest.ckpt -o out/drinking_none
echo drinking state predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/DrinkingJaco_1kk/latest.ckpt -o out/drinking_state_predictor --safety_model drinking_state_predictor
echo drinking vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/DrinkingJaco_1kk/latest.ckpt -o out/drinking_vae --safety_model drinking_vae

echo arm_manipulation
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ArmManipulationJaco_2.5kk/latest.ckpt -o out/arm_manipulation_none
echo arm_manipulation state predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ArmManipulationJaco_2.5kk/latest.ckpt -o out/arm_manipulation_state_predictor --safety_model arm_manipulation_state_predictor
echo arm_manipulation vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ArmManipulationJaco_2.5kk/latest.ckpt -o out/arm_manipulation_vae --safety_model arm_manipulation_vae

echo scratch_itch
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ScratchItchJaco_600k/latest.ckpt -o out/scratch_itch_none
echo scratch_itch state predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ScratchItchJaco_600k/latest.ckpt -o out/scratch_itch_state_predictor --safety_model scratch_itch_state_predictor
echo scratch_itch vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ScratchItchJaco_600k/latest.ckpt -o out/scratch_itch_vae --safety_model scratch_itch_vae

echo bed_bathing
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/BedBathing_2.5kk/latest.ckpt -o out/bed_bathing_none
echo bed_bathing state_predictor
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/BedBathing_2.5kk/latest.ckpt -o out/bed_bathing_state_predictor --safety_model bed_bathing_state_predictor
echo bed_bathing vae
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/BedBathing_2.5kk/latest.ckpt -o out/bed_bathing_vae --safety_model bed_bathing_vae
