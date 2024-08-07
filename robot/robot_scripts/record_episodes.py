import os
import time
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pynput.keyboard import Key, Listener

from constants import DT, START_ARM_POSE, TASK_CONFIGS, ROBOT_MODEL, NUM_JOINTS
from constants import MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE, PUPPET_GRIPPER_JOINT_OPEN
from robot_utils import Recorder, ImageRecorder, get_arm_gripper_positions
from robot_utils import move_arms, torque_on, torque_off, move_grippers
from real_env import make_real_env, get_action

from interbotix_xs_modules.arm import InterbotixManipulatorXS

import IPython
e = IPython.embed


def opening_ceremony(master_bot, puppet_bot):
    """ Move all 2 robots to a pose where it is easy to start demonstration """
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(puppet_bot)
    torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:NUM_JOINTS]
    move_arms([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1.5)
    # move grippers to starting position
    move_grippers([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if (gripper_pos < close_thresh) and (gripper_pos < close_thresh):
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot)
    print(f'Started!')


def capture_one_episode(dt, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite):
    print(f'Dataset name: {dataset_name}')

    # source of data
    master_bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name="arm", gripper_name="gripper",
                                         robot_name=f'master', init_node=True)
    shapes = dict()
    env = make_real_env(init_node=False, setup_robots=False)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    # move 2 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(master_bot, env.puppet_bot)

    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []

    epi_length = max_timesteps
    epi_length_recorded = False


    class KeyboardCback:
        def __init__(self):
            self.pressed_keys = set()

        def on_release(self, key):
            self.pressed_keys.add(key)
            if key == Key.space:
                return False
        
        def is_space_pressed(self):
            if Key.space in self.pressed_keys:
                self.pressed_keys.remove(Key.space)
                return True
            return False

    cback = KeyboardCback()
    # Collect events until released
    with Listener(
            on_release=cback.on_release) as listener:

        for t in tqdm(range(max_timesteps)):
            t0 = time.time() #
            action = get_action(master_bot)
            t1 = time.time() #
            ts = env.step(action)
            t2 = time.time() #
            timesteps.append(ts)
            actions.append(action)
            actual_dt_history.append([t0, t1, t2])

            if cback.is_space_pressed() and not epi_length_recorded:
               epi_length = len(timesteps) 
               epi_length_recorded = True
               print(f"Episode length recorded (t={t}): {epi_length}")

        listener.join()

    # Torque on both master bots
    torque_on(master_bot)
    # Open puppet grippers
    move_grippers([env.puppet_bot], [PUPPET_GRIPPER_JOINT_OPEN], move_time=0.5)

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 42:
        return False

    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
    - qpos                  (6,)          'float64'
    - qvel                  (6,)          'float64'
    
    action                  (6,)          'float64'
    """

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/length': [epi_length],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)

        shapes['/observations/qpos'] = (len(ts.observation['qpos']),)
        shapes['/observations/qvel'] = (len(ts.observation['qvel']),)
        shapes['/observations/effort'] = (len(ts.observation['effort']),)
        shapes['/action'] = (len(action),)

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            shapes[f'/observations/images/{cam_name}'] = ts.observation['images'][cam_name].shape

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            shape = shapes[f'/observations/images/{cam_name}']
            _ = image.create_dataset(cam_name, (max_timesteps, *shape), dtype='uint8',
                                     chunks=(1, *shape), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        _ = obs.create_dataset('qpos', (max_timesteps, *shapes['/observations/qpos']))
        _ = obs.create_dataset('qvel', (max_timesteps, *shapes['/observations/qvel']))
        _ = obs.create_dataset('effort', (max_timesteps, *shapes['/observations/effort']))
        _ = root.create_dataset('action', (max_timesteps, *shapes['/action']))
        _ = root.create_dataset('length', (1,))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


def main(args):
    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    while True:
        is_healthy = capture_one_episode(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite)
        if is_healthy:
            break


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(f'Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}')
    return freq_mean

def debug():
    print(f'====== Debug mode ======')
    recorder = Recorder('main', is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args()))
    # debug()


