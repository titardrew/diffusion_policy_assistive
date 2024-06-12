"""
Usage:
python assistive_dp_demo.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from typing import Optional

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
from safety_model.train import *

import assistive_gym


def set_camera_pos(env, cam_id):
    env.unwrapped.render_camera_width = 1920
    env.unwrapped.render_camera_height = 1080

    if cam_id == 0:
        # cam 1
        cam_target = np.array([-0.2, 0.0, 0.75])
        cam_eye = np.array([0.5, -0.75, 1.5])
    else:
        # cam 2
        cam_target = np.array([+0.2, -0.2, 0.85])
        cam_eye = np.array([-0.5, -0.35, 1.5])

    env.unwrapped.render_camera_target = cam_target
    env.unwrapped.render_camera_eye = cam_eye
    # hack
    if hasattr(env.unwrapped, '_camera_auto_setup'):
        del env.unwrapped._camera_auto_setup


def make_env(env_name, seed, n_obs_steps, n_action_steps, max_steps, safety_model_horizon, gui=True):
    env = gym.make('assistive_gym:' + env_name)
    env.seed(seed)

    # this script requires GUI.
    if gui:
        env.render()
    else:
        set_camera_pos(env, 0)

    return MultiStepWrapper(
        env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        horizon=safety_model_horizon,
    )



def demo_checkpoint(
        checkpoint: str,
        output_dir: str,
        device: str,
        seed: int,
        safety_model: Optional[str] = None,
        safety_model_overrides: Optional[dict] = None
):
    if safety_model:
        if safety_model.endswith(".pth"):
            safety_model = torch.load(safety_model)
            safety_model.override_params({
                "use_reconstruction_loss": False,
                "reconstruction_threshold": 4.0,
                "use_pw_kl": True,
                "pw_kl_threshold": 0.0006,
            })
        else:
            safety_model_path, safety_model_kwargs = get_registered_safety_model_file_and_params(safety_model)
            safety_model = torch.load(safety_model_path)
            safety_model.override_params(safety_model_kwargs)

        if safety_model_overrides:
            safety_model.override_params(safety_model_overrides)

        safety_model.to(torch.device(device))

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
        safety_model_horizon=safety_model.horizon if safety_model else 2,
        gui=True,
    )

    policy.n_action_steps = 8
    rr.init("Assistive twin", spawn=True)
    #rr.save("Assistive.rrd")

    total_reward = 0
    force_list = []
    task_success = 0.0
    done = False

    obs = env.reset()

    while not done:
        np_obs_dict = {'obs': obs.astype(np.float32)[None, ...],}
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        n_input_obs = obs.shape[0]

        set_camera_pos(env, cam_id=0)
        rgb = env.render(mode="rgb_array")
        rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=95))

        set_camera_pos(env, cam_id=1)
        rgb = env.render(mode="rgb_array")
        rr.log("camera/image2", rr.Image(rgb).compress(jpeg_quality=95))

        np_horizon_obs_dict = {'obs': env.cached_obs.astype(np.float32)[None, ...],}
        np_horizon_actions_dict = {'action': env.cached_actions.astype(np.float32)[None, ...],}
        horizon_obs_dict = dict_apply(np_horizon_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
        horizon_actions_dict = dict_apply(np_horizon_actions_dict, lambda x: torch.from_numpy(x).to(device=device))

        def get_action():
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action'][0]
            action_pred = np_action_dict['action_pred'][0]

            # print(f"T_o: {policy.n_obs_steps}")
            # print(f"T_a: {policy.n_action_steps}")
            # print(f"H: {policy.horizon}")
            # print(f"obs: {obs_dict['obs'].shape}")
            # print(f"action: {action_dict['action_pred'].shape}")

            if safety_model:
                if isinstance(safety_model, EnsembleStatePredictionSM):
                    batch = {
                        'obs': horizon_obs_dict['obs'],
                        'action': action_dict['action_pred'],
                    }
                elif isinstance(safety_model, VariationalAutoEncoderSM):
                    if safety_model.in_act_horizon > safety_model.in_obs_horizon:
                        n_obs_steps = cfg.task.env_runner.n_obs_steps
                        n_action_steps = cfg.task.env_runner.n_action_steps
                        action_vec = torch.cat([horizon_actions_dict['action'], action_dict['action_pred'][:, n_obs_steps:]], dim=1)
                    else:
                        action_vec = horizon_actions_dict['action']
                    batch = {
                        'obs': horizon_obs_dict['obs'],
                        'action': action_vec,
                    }
                is_safe, stats = safety_model.check_safety_runner(batch, rr_log=True, return_stats=True)
            return action


        action = get_action()

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
        
# 79

if __name__ == '__main__':
    DEFAULT_CHECKPOINT = '/home/aty/Data/TeleopDPCheckpoints/FeedingJaco-v1/horizon_10_n_obs_1_n_act_2/checkpoints/best.ckpt'

    class ParseKVAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for each in values:
                try:
                    key, value = each.split("=")
                    getattr(namespace, self.dest)[key] = value
                except ValueError as ex:
                    message = "\nTraceback: {}".format(ex)
                    message += "\nError on '{}' || It should be 'key=value'".format(each)
                    raise argparse.ArgumentError(self, str(message))


    parser = argparse.ArgumentParser(description='Use a pretrained diffusion policy to play.')
    parser.add_argument(
        '--checkpoint',
        help='Path to the diffusion policy checkpoint. It will pick up the env name and other details.',
        default=DEFAULT_CHECKPOINT)
    parser.add_argument('--seed', type=int, help='Random env seed.')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--safety_model') #, default="ensemble_state_prediction_sm3.pth")
    parser.add_argument(
        "--safety_model_overrides",
        nargs='+',
        action=ParseKVAction,
        help='--safety_model_overrides key1=val1 key2=val2',
        metavar="KEY1=VALUE1",
    )

    parser.add_argument('-o', '--output_dir', default="/tmp/output_dir_for_diffusion_policy/")
    args = parser.parse_args()

    demo_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        safety_model=args.safety_model,
        safety_model_overrides=args.safety_model_overrides,
    )