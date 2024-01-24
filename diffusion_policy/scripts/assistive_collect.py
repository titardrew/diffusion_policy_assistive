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

from diffusion_policy.common.replay_buffer import ReplayBuffer

def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.PPOConfig()
        config.train_batch_size = 19200
        config.num_sgd_iter = 50
        config.sgd_minibatch_size = 128
        config.lambda_ = 0.95
        config.model['fcnet_hiddens'] = [100, 100]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.SACConfig()
        config.min_train_timesteps_per_iteration = 400
        config.learning_starts = 1000
        config.q_model_config['fcnet_hiddens'] = [100, 100]
        config.policy_model_config['fcnet_hiddens'] = [100, 100]
    config.num_rollout_workers = num_processes
    config.num_cpus_per_worker = 0
    config.seed = seed
    # config.log_level = 'ERROR'
    if coop:
        obs, _ = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config.multiagent
        config.multiagent = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config.env_config = {'num_agents': 2}
    return {**config, **extra_configs}


def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = Algorithm.from_checkpoint(policy_path)
        #agent = ppo.PPO(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
        #agent = ppo.PPO(ppo.PPOConfig(), 'assistive_gym:' + env_name)
    elif algo == 'sac':
        agent = Algorithm.from_checkpoint(policy_path)
        #agent = sac.SAC(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
        #agent = sac.SAC(sac.SACConfig(), 'assistive_gym:' + env_name)
    if policy_path != '':
        #agent.restore(policy_path)
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
        n_episodes_render=0,
        n_episodes=100,
        seed=0,
        verbose=False,
        extra_configs={}
):
    coop = False

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop, seed=seed)

    if n_episodes_render > 0:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 10
        cam_w = 1920 // 4
        cam_h = 1080 // 4
        video_path = Path(output_path).parent / (Path(output_path).stem + "_rec.mp4")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (cam_w, cam_h))
        env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=cam_w, camera_height=cam_h)

    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    buffer = ReplayBuffer.create_empty_numpy()
    rewards = []
    forces = []
    task_successes = []
    n_collected_episodes = 0
    n_tries = 0
    n_failures = 0
    n_successes = 0
    print(f"----------------------")
    while n_collected_episodes < n_episodes:
        n_tries += 1
        if n_tries >= 50:
            raise RuntimeError("Policy is garbage. 50 failures straight.")

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

            if n_episodes_render > 0:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                print(img.shape)
                writer.write(img[..., :3])


        if task_success > 0.0:
            n_collected_episodes += 1
            buffer.add_episode({
                'state': np.array(state_history),
                'action': np.array(action_history),
                'reward': np.array(reward_history),
            })
            print(f"Collected: {n_collected_episodes}/{n_episodes}")
            print(f"Episode len: {len(state_history)}")
            print(f"Tries: {n_tries}")
            print(f"----------------------")
            n_failures += n_tries - 1
            n_successes += 1
            n_tries = 0
        if n_episodes_render > 0:
            n_episodes_render -= 1
            if n_episodes_render == 0:
                writer.release()
                mb = (video_path.stat().st_size / 10**6)
                print(f"[Viz] Rendered all. Filesize: {mb} Mb.")
                print(f"----------------------")
        #print(reward_total)
        rewards.append(reward_total)
        forces.append(np.mean(force_list))
        task_successes.append(task_success)
        if verbose:
            print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
        sys.stdout.flush()
    env.disconnect()


    buffer.save_to_path(output_path, chunk_length=-1)

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Forces:', forces)
    print('Force Mean:', np.mean(forces))
    print('Force Std:', np.std(forces))

    print('Task Failure Rate (%):', n_failures / (n_successes + n_failures) * 100)
    # print('Task Successes:', task_successes)
    print('Task Success Mean:', np.mean(task_successes))
    print('Task Success Std:', np.std(task_successes))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of collected episodes (default: 100)')
    parser.add_argument('--output_path', default='./ppo.zarr',
                        help='Directory to save trained policy in (default ./ppo.zarr)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')

    parser.add_argument('--env', default='ScratchItchJaco-v1',
                        help='Environment to train on (default: ScratchItchJaco-v1)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--render-episodes', type=int, default=0,
                        help='How many episodes to render')
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
        n_episodes_render=args.render_episodes,
        n_episodes=args.episodes,
        seed=args.seed,
        verbose=args.verbose,
    )