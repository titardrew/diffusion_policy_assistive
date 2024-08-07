from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import move_arms, torque_on, torque_off
from constants import ROBOT_MODEL, SLEEP_ARM_POSE
import argparse

def main(do_master, do_puppet):
    bots = []
    poses = []
    init_node = True
    if do_master:
        bots.append(InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name="arm", gripper_name="gripper", robot_name=f'master', init_node=init_node))
        poses += [SLEEP_ARM_POSE]
        init_node = False
    if do_puppet:
        bots.append(InterbotixManipulatorXS(robot_model=ROBOT_MODEL, group_name="arm", gripper_name="gripper", robot_name=f'puppet', init_node=init_node))
        poses += [SLEEP_ARM_POSE]
        init_node = False

    for bot in bots:
        torque_on(bot)

    move_arms(bots, poses, move_time=2)
    
    for bot in bots:
        torque_off(bot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--master", action="store_true")
    parser.add_argument("-p", "--puppet", action="store_true")
    args = parser.parse_args()
    assert args.master or args.puppet
    main(args.master, args.puppet)
