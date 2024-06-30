set -ex

export PYTHONPATH="."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

export MS_ARGS="--num_epochs 100 --full_episode --in_horizon 0 --gap_horizon 0 --out_obs_horizon 0 --model_type mvt_flow --ensemble_size 1"
python safety_model/train.py \
    --test_parquet_path parquet_datasets/feeding_250_test.parquet \
    --zarr_path teleop_datasets/FeedingJaco-v1.zarr/ \
    --test_zarr_path out_voraus/feeding_250/zarr_recording/cat.zarr/ \
    --save_path test_mvt_flow.pth \
    --env_type Feeding \
    --metric loss \
    --lr 8e-4 \
    --batch_size 32 \
    --test_freq 5 \
    --backend tb \
    --experiment_path runs/test_mvt_flow_feeding_grad \
    --device cuda \
    $MS_ARGS


# 
# [x] 1. Check the pipeline. It must work properly datawise.
# [x] 2. Try different uncertainty metric, i.e. max variance norm.
# [x] 3. Check if data is actually normalized properly.
# [x] 4. Do something with the overfitting. Why Groupnorm does not work? Maybe layernorm? -- Dropout worked!
# [x] 5. Possibly UnetConv1D.
# [x] 6. Add time component.
# [x] 7. Try state prediction - state prediction error.
# [ ] 8. Try VAE approach, conv-deconv net, possibly UnetConv1D.
# [ ] 9. Try attention and transformers.
# [ ] 10. CVAE for state-prediction.
#