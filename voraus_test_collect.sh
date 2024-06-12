#!/bin/bash

set -ex

out="out_voraus"
mkdir -p $out

# feeding 250
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt -o ${out}/feeding_250
# feeding 100
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1-100/horizon_10_n_obs_2_n_act_2/checkoint/best.ckpt -o ${out}/feeding_100
# feeding 50
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1-50/horizon_10_n_obs_2_n_act_2/checkpoint/best.ckpt -o ${out}/feeding_50
# drinking
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/DrinkingJaco_1kk/latest.ckpt -o ${out}/drinking
# arm_manipulation
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ArmManipulationJaco_2.5kk/latest.ckpt -o ${out}/arm_manipulation
# scratch_itch
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/ScratchItchJaco_2.5kk/latest.ckpt -o ${out}/scratch_itch
# bed_bathing
HYDRA_FULL_ERROR=1 PYTHONPATH="." python eval.py -y -z -d cuda --n_tests 200 -c /home/aty/Data/AssistiveDiffusion/Experiments/BedBathing_2.5kk/latest.ckpt -o ${out}/bed_bathing