"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from safety_model.train import *

from omegaconf import open_dict

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-z', '--record_zarr', is_flag=True, show_default=True, default=False)
@click.option('-sm', '--safety_model', default=None, required=False)
@click.option('-m', '--safety_model_metric', default=None, required=False)
@click.option('-n', '--n_tests', default=50, required=False)
@click.option("-y", '--exists_ok', is_flag=True, show_default=True, default=False)
def main(checkpoint, output_dir, device, record_zarr=False, safety_model=None, safety_model_metric=None, n_tests=50, exists_ok=False):
    if os.path.exists(output_dir) and (not exists_ok):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    if safety_model:
        safety_model_path, safety_model_kwargs = get_registered_safety_model_file_and_params(safety_model, safety_model_metric)
        with open_dict(cfg):
            cfg.task.env_runner.safety_model_path = safety_model_path
            cfg.task.env_runner.safety_model_kwargs = safety_model_kwargs

    if record_zarr:
        with open_dict(cfg):
            cfg.task.env_runner.record_zarr = record_zarr

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_test = n_tests
    cfg.task.env_runner.n_envs = 50 if n_tests > 50 else None
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    import numpy as np
    json_log = dict()
    from collections.abc import Iterable

    def to_jsonable(a):
        if isinstance(a, np.integer):
            return int(a)
        elif isinstance(a, np.bool_):
            return bool(a)
        elif isinstance(a, np.floating):
            return float(a)
        else:
            return a

    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, dict):
            json_log[key] = {k: to_jsonable(v) for k, v in value.items()}
        elif isinstance(value, Iterable):
            json_log[key] = [to_jsonable(a) for a in list(value)]
        else:
            json_log[key] = to_jsonable(value)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
