import os, sys, multiprocessing, ray, shutil, argparse, importlib, glob
from pathlib import Path
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.algorithms import ppo, sac
from ray.rllib.algorithms.algorithm import Algorithm

from ray.rllib.utils import check_env
from ray.tune.logger import pretty_print
#import ray.rllib.core
from numpngw import write_apng
import assistive_gym

import rerun as rr


def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo' or algo == 'sac':
        agent = Algorithm.from_checkpoint(policy_path)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
            print("restored!")
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                print("restored!")
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def draw_state(env_name, obs, info, reward, done):
    if "FeedingJaco" in env_name:
        success_percent = info['task_success_percent']
        rr.log('info/success_percent', rr.TimeSeriesScalar(scalar=success_percent))
        verbose_obs = info['verbose_obs']
        robot_links = info['robot_links']
        link_strip = []
        for link_pos, link_orient in robot_links:
            link_strip.append(list(link_pos))

        rr.log('world/robot', rr.LineStrips3D(
            strips=[link_strip]
        ))
    else:
        raise NotImplementedError()


def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:' + env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    check_env(env)
    return env


def collect_with_policy(
        output_path,
        env_name,
        algo,
        policy_path,
        n_episodes=100,
        min_reward=-float("inf"),
        seed=0,
        verbose=False,
        extra_configs={}
):
    coop = False

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed)

    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    Path(output_path).mkdir(exist_ok=True, parents=True)

    rr.init("assistive_rerun")

    rewards = []
    forces = []
    task_successes = []
    n_collected_episodes = 0
    n_tries = 0
    n_failures = 0
    n_successes = 0
    print(f"----------------------")
    n_episodes_total = 0
    while n_collected_episodes < n_episodes:
        n_episodes_total += 1
        n_tries += 1
        if n_tries >= 1000:
            raise RuntimeError(f"Policy is garbage. {n_tries} failures straight.")

        rr.save(str(Path(output_path) / f"episode_{n_episodes_total}.rrd"))

        obs = env.reset()
        done = False
        reward_total = 0.0
        force_list = []
        task_success = 0.0
        state_history = []
        action_history = []
        reward_history = []
        while not done:
            if coop:
                raise NotImplementedError("coop not used atm.")
                """
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                reward = reward['robot']
                done = done['__all__']
                info = info['robot']
                """
            else:
                action = test_agent.compute_single_action(obs)
                state_history.append(obs)
                action_history.append(action)
                obs, reward, done, info = env.step(action)
                reward_history.append(reward)

            reward_total += reward
            force_list.append(info['total_force_on_human'])
            task_success = info['task_success']

            # Capture (render) an image from the camera
            rgb = env.render(mode="rgb_array")
            rr.log("camera/rgb", rr.Image(rgb).compress())


        if task_success > 0.0 and reward_total >= min_reward:
            n_collected_episodes += 1
            print(f"Collected: {n_collected_episodes}/{n_episodes}")
            print(f"Episode len: {len(state_history)}")
            print(f"Tries: {n_tries}")
            print(f"----------------------")
            n_failures += n_tries - 1
            n_successes += 1
            n_tries = 0
        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    env.disconnect()


    print('\n', '-'*50, '\n')
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    print('Task Collection Rate (%):', n_successes / (n_successes + n_failures) * 100)
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of collected episodes (default: 100)')
    parser.add_argument('--output_path', default='./assisitve_reruns/',
                        help='Directory to save rerun file')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')

    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to train on (default: ScratchItchJaco-v1)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(exist_ok=True)

    #coop = ('Human' in args.env)
    coop = False
    checkpoint_path = None

    collect_with_policy(
        output_path=args.output_path,
        env_name=args.env,
        algo=args.algo,
        policy_path=args.load_policy_path,
        n_episodes=args.episodes,
        seed=args.seed,
        verbose=args.verbose,
    )