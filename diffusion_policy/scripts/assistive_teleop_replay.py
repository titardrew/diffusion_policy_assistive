from pathlib import Path
import sys, argparse, importlib

from pathlib import Path
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import zarr

import assistive_gym


def make_env(env_name, seed, coop=False):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    # teleop requires GUI.
    env.render()
    return env


def replay_teleop(zarr_path: Path, env_name: str, seed: int):
    env = make_env(env_name, seed)
    store = zarr.open(str(zarr_path))
    actions = store.data.action[:]
    total_reward = 0
    force_list = []
    task_success = 0.0
    env.reset()
    for i_step, action in enumerate(actions):
        _, reward, done, info = env.step(action)

        force_list.append(info['total_force_on_human'])
        task_success = info['task_success']
        total_reward += reward

        if done:
            break

    print(f"Done on step #{i_step + 1}/{len(actions)}. Reward: {total_reward}, Force: {np.mean(force_list)}")
    if task_success > 0.0:
        print(f"SUCCESS!")
    else:
        print(f"FAILED!")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay a teleop recording.')
    parser.add_argument('--input', help='Path to zarr with an episode recording')
    parser.add_argument('--env', default='FeedingJaco-v1',
                        help='Environment to train on (default: FeedingJaco-v1')
    parser.add_argument('--seed', type=int, help='Random seed. Important!')
    args = parser.parse_args()

    assert args.seed is not None

    replay_teleop(
        zarr_path=Path(args.input),
        env_name=args.env,
        seed=args.seed,
    )