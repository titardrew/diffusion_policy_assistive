import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import dm_env

from constants import DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE, ROBOT_MODEL, NUM_JOINTS
from robot_utils import Recorder, ImageRecorder
from robot_utils import setup_master_bot, setup_puppet_bot, move_arms, move_grippers
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

import IPython
e = IPython.embed

class RealEnv:
    """
    Environment for real robot manipulation
    Action space:      [arm_qpos (5),             # absolute joint position
                        gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                       ]

    Observation space: {"qpos": Concat[ arm_qpos (5),          # absolute joint position
                                        gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                      ]
                        "qvel": Concat[ arm_qvel (5),          # absolute joint velocity (rad)
                                        gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                      ]
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                  }
    """

    def __init__(self, init_node, setup_robots=True):
        self.puppet_bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name="arm", gripper_name="gripper",
                                                  robot_name=f'puppet', init_node=init_node)
        if setup_robots:
            self.setup_robots()

        self.recorder = Recorder('main', init_node=False)
        self.image_recorder = ImageRecorder(init_node=False)
        self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        setup_puppet_bot(self.puppet_bot)

    def get_qpos(self):
        qpos_raw = self.recorder.qpos
        arm_qpos = qpos_raw[:NUM_JOINTS]
        gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[NUM_JOINTS+1])] # this is position not joint
        return np.concatenate([arm_qpos, gripper_qpos])

    def get_qvel(self):
        qvel_raw = self.recorder.qvel
        arm_qvel = qvel_raw[:NUM_JOINTS]
        gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[NUM_JOINTS+1])]
        return np.concatenate([arm_qvel, gripper_qvel])

    def get_effort(self):
        effort_raw = self.recorder.effort
        robot_effort = effort_raw[:NUM_JOINTS+1]
        return robot_effort

    def get_images(self):
        return self.image_recorder.get_images()

    def set_gripper_pose(self, gripper_desired_pos_normalized):
        gripper_desired_joint = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(gripper_desired_pos_normalized)
        self.gripper_command.cmd = gripper_desired_joint
        self.puppet_bot.gripper.core.pub_single.publish(self.gripper_command)

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:NUM_JOINTS]
        move_arms([self.puppet_bot], [reset_position], move_time=1)

    def _reset_gripper(self):
        """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
        move_grippers([self.puppet_bot], [PUPPET_GRIPPER_JOINT_OPEN], move_time=0.5)
        move_grippers([self.puppet_bot], [PUPPET_GRIPPER_JOINT_CLOSE], move_time=1)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot puppet robot gripper motors
            self.puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())

    def step(self, action):
        self.puppet_bot.arm.set_joint_positions(action[:NUM_JOINTS], blocking=False)
        self.set_gripper_pose(action[-1])
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())


def get_action(master_bot):
    action = np.zeros(NUM_JOINTS + 1)  # NUM_JOINTS joint + 1 gripper
    # Arm actions
    action[:NUM_JOINTS] = master_bot.dxl.joint_states.position[:NUM_JOINTS]
    # Gripper
    action[NUM_JOINTS] = MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot.dxl.joint_states.position[NUM_JOINTS])
    return action


def make_real_env(init_node, setup_robots=True):
    env = RealEnv(init_node, setup_robots)
    return env


def test_real_teleop():
    """
    Test teleoperation and show image observations onscreen.
    It first reads joint poses from the master arm.
    Then use it as action to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleoperation and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """

    onscreen_render = True
    render_cam = 'cam_high'

    # source of data
    master_bot = InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name="arm", gripper_name="gripper",
                                         robot_name=f'master', init_node=True)
    setup_master_bot(master_bot)

    # setup the environment
    env = make_real_env(init_node=False)
    ts = env.reset(fake=True)
    episode = [ts]
    # setup visualization
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam])
        plt.ion()

    for t in range(1000):
        action = get_action(master_bot)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == '__main__':
    test_real_teleop()