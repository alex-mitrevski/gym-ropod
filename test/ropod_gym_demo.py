import sys
from os.path import join
import time
import signal

import gym
import rospkg
from termcolor import colored

from gym_ropod.envs.ropod_nav_env import RopodNavActions

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    ropod_sim_pkg_path = rospack.get_path('ropod_sim_model')
    launch_file = join(ropod_sim_pkg_path, 'launch/simulator/gazebo_simulator.launch')
    number_of_steps = 500

    env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file)
    env.render(mode='human')
    time.sleep(5)
    env.reset()

    try:
        print(colored('Running simulation for {0} steps'.format(number_of_steps), 'green'))
        for i in range(number_of_steps):
            action = env.action_space.sample()
            print(colored('Sending command "{0}"'.format(RopodNavActions.action_num_to_str[action]), 'green'))
            env.step(action)
            time.sleep(0.05)

        env.reset()
        signal.pause()
    except KeyboardInterrupt:
        print(colored('Closing simulator', 'green'))
        env.close()
        sys.exit()
