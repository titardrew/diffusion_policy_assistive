import sys, argparse, importlib

from pathlib import Path
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import pybullet as p

import assistive_gym

from diffusion_policy.common.replay_buffer import ReplayBuffer


def make_env(env_name, coop=False, seed=1001):
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


# Map keys to position and orientation end effector movements
POS_KEYS_ACTIONS = {ord('j'): np.array([-0.01, 0, 0]), ord('l'): np.array([0.01, 0, 0]),
                    ord('u'): np.array([0, -0.01, 0]), ord('o'): np.array([0, 0.01, 0]),
                    ord('k'): np.array([0, 0, -0.01]), ord('i'): np.array([0, 0, 0.01])}
RPY_KEYS_ACTIONS = {ord('k'): np.array([-0.05, 0, 0]), ord('i'): np.array([0.05, 0, 0]),
                    ord('u'): np.array([0, -0.05, 0]), ord('o'): np.array([0, 0.05, 0]),
                    ord('j'): np.array([0, 0, -0.05]), ord('l'): np.array([0, 0, 0.05])}


def collect_with_teleop(
        output_path,
        env_name,
        render=False,
        seed=0,
):
    coop = False

    env = make_env(env_name, coop, seed=seed)

    if render:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = 10
        cam_w = 1920 // 4
        cam_h = 1080 // 4
        video_path = Path(output_path).parent / (Path(output_path).stem + "_rec.mp4")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (cam_w, cam_h))

    buffer = ReplayBuffer.create_empty_numpy()
    print(f"----------------------")

    obs = env.reset()

    # needed for teleop
    start_pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    start_rpy = env.get_euler(orient)
    target_pos_offset = np.zeros(3)
    target_rpy_offset = np.zeros(3)

    done = False
    reward_total = 0.0
    force_list = []
    task_success = 0.0
    state_history = []
    action_history = []
    reward_history = []
    while not done:
        keys = p.getKeyboardEvents()
        # Process position movement keys ('u', 'i', 'o', 'j', 'k', 'l')
        for key, action in POS_KEYS_ACTIONS.items():
            if p.B3G_SHIFT not in keys and key in keys and keys[key] & p.KEY_IS_DOWN:
                target_pos_offset += action
        # Process rpy movement keys (shift + movement keys)
        for key, action in RPY_KEYS_ACTIONS.items():
            if p.B3G_SHIFT in keys and keys[p.B3G_SHIFT] & p.KEY_IS_DOWN and (key in keys and keys[key] & p.KEY_IS_DOWN):
                target_rpy_offset += action

        # print('Target position offset:', target_pos_offset, 'Target rpy offset:', target_rpy_offset)
        target_pos = start_pos + target_pos_offset
        target_rpy = start_rpy + target_rpy_offset

        # Use inverse kinematics to compute the joint angles for the robot's arm
        # so that its end effector moves to the target position.
        target_joint_angles = env.robot.ik(env.robot.right_end_effector, target_pos, env.get_quaternion(target_rpy), env.robot.right_arm_ik_indices, max_iterations=200, use_current_as_rest=True)
        # Get current joint angles of the robot's arm
        current_joint_angles = env.robot.get_joint_angles(env.robot.right_arm_joint_indices)
        # Compute the action as the difference between target and current joint angles.
        action = (target_joint_angles - current_joint_angles) * 10

        state_history.append(obs)
        action_history.append(action)
        obs, reward, done, info = env.step(action)
        reward_history.append(reward)

        reward_total += reward
        force_list.append(info['total_force_on_human'])
        task_success = info['task_success']

        if render:
            # Capture (render) an image from the camera
            rgb = env.render(mode="rgb_array")
            writer.write(rgb)


    if task_success > 0.0:
        buffer.add_episode({
            'state': np.array(state_history),
            'action': np.array(action_history),
            'reward': np.array(reward_history),
        })
        print(f"Collected")
        print(f"Episode len: {len(state_history)}")
        print(f"----------------------")
    else:
        print(f"Not collected!!!")
        print(f"----------------------")
    if render:
        writer.release()
        mb = (video_path.stat().st_size / 10**6)
        print(f"[Viz] Rendered all. Filesize: {mb} Mb.")
        print(f"----------------------")
    print('Reward total: %.2f, mean force: %.2f, task success: %r' % (reward_total, np.mean(force_list), task_success))
    env.disconnect()

    if done and len(state_history) < 200:
        buffer.save_to_path(output_path, chunk_length=-1)
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use teleoperation to collect AssistiveGym trajectories.')
    parser.add_argument('--output_dir', default='./teleop_episodes',
                        help='Directory to save trained policy in (default ./teleop_episodes)')
    parser.add_argument('--env', default='FeedingJaco-v1',
                        help='Environment to train on (default: FeedingJaco-v1)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    parser.add_argument('--render', action="store_true",
                        help='Whether to save a video.')
    args = parser.parse_args()


    if args.seed is None:
        import random
        args.seed = random.randint(0, 99999999)
    output_dir = (Path(args.output_dir) / args.env)
    output_dir.mkdir(parents=True, exist_ok=True)

    nums = list(int(episode_fname.stem.split("_")[1]) for episode_fname in output_dir.glob("episode_*.zarr"))
    num = max(nums) + 1 if nums else 1
    output_path = Path(output_dir) / f"episode_{num:05d}.zarr"

    collect_with_teleop(
        output_path=output_path,
        env_name=args.env,
        render=args.render,
        seed=args.seed,
    )