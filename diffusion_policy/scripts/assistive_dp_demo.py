"""
Usage:
python assistive_dp_demo.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import argparse
import os
import pathlib
import click
import hydra
import torch
import dill

import rerun as rr

try:
    import gymnasium as gym
except ImportError:
    import gym

import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.common.pytorch_util import dict_apply

import assistive_gym


def make_env(env_name, seed, n_obs_steps, n_action_steps, max_steps):
    env = gym.make('assistive_gym:' + env_name)
    env.seed(seed)

    # this script requires GUI.
    env.render()

    return MultiStepWrapper(
        env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )


def demo_checkpoint(checkpoint, output_dir, device, seed):

    #if os.path.exists(output_dir):
    #    click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    print(
        "\n".join([
            "==========================================",
            "MODEL DETAILS:",
            f"\tcheckpoint: {checkpoint}",
            f"\tn_obs_steps: {cfg.task.env_runner.n_obs_steps}",
            f"\tn_action_steps: {cfg.task.env_runner.n_action_steps}",
            f"\thorizon: {cfg.task.dataset.horizon}",
            f"\tdataset (zarr) path: {cfg.task.dataset.zarr_path}",
            "==========================================",
        ])
    )

    env = make_env(
        env_name=f"{cfg.task.task_name}{cfg.task.robot_name}-v1",
        seed=seed,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        max_steps=200,
    )

    rr.init("Assistive twin", spawn=True)

    total_reward = 0
    force_list = []
    task_success = 0.0
    done = False

    obs = env.reset()

    while not done:
        rr.log("random_scalar", rr.Scalar(np.random.randn()))
        np_obs_dict = {
            'obs': obs.astype(np.float32)[None, ...],
        }

        obs_dict = dict_apply(np_obs_dict, 
            lambda x: torch.from_numpy(x).to(
                device=device))

        # run policy
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        # device_transfer
        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        action = np_action_dict['action'][0]
        action_pred = np_action_dict['action_pred'][0]
        breakpoint()

        # step env
        obs, reward, done, info = env.step(action)

        force_list.append(info['total_force_on_human'])
        task_success = np.mean(info['task_success'])
        total_reward += reward

    print(f"Reward: {total_reward}, Force: {np.mean(force_list)}")
    if task_success > 0.0:
        print(f"SUCCESS!")
    else:
        print(f"FAILED!")
    
    rr.disconnect()
        

if __name__ == '__main__':
    DEFAULT_CHECKPOINT = '/home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt'

    parser = argparse.ArgumentParser(description='Use a pretrained diffusion policy to play.')
    parser.add_argument(
        '--checkpoint',
        help='Path to the diffusion policy checkpoint. It will pick up the env name and other details.',
        default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seed', type=int, help='Random env seed.')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('-o', '--output_dir', default="/tmp/output_dir_for_diffusion_policy/")
    args = parser.parse_args()

    demo_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )