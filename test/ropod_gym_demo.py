import sys
from os.path import join
import time
import signal

import gym
import rospkg
from termcolor import colored

from gym_ropod.envs.ropod_nav_env import RopodNavActions

RUNNING = False

def sigint_handler(signum, frame):
    global RUNNING
    print(colored('Simulation interupted', 'red'))
    RUNNING = False

def main():
    global RUNNING
    rospack = rospkg.RosPack()
    ropod_sim_pkg_path = rospack.get_path('ropod_sim_model')
    launch_file = join(ropod_sim_pkg_path, 'launch/simulator/gazebo_simulator.launch')
    number_of_steps = 500

    env = gym.make('ropod-nav-discrete-v0', launch_file_path=launch_file)
    env.render(mode='human')
    time.sleep(5)
    env.reset()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        print(colored('Running simulation for {0} steps'.format(number_of_steps), 'green'))
        episode_step_count = 0
        RUNNING = True
        for i in range(number_of_steps):
            if not RUNNING:
                break
            action = env.action_space.sample()
            (goal, obs, reward, done) = env.step(action)
            print(colored('Step {0}: "{1}" -> reward {2}'.format(i, RopodNavActions.action_num_to_str[action],
                                                       reward), 'green'))
            episode_step_count += 1
            if done:
                print(colored('Episode done after {0} steps'.format(episode_step_count), 'yellow'))
                print(colored('Resetting environment', 'yellow'))
                env.reset()
                episode_step_count = 0
            else:
                time.sleep(0.05)
    except Exception as e:
        print(colored('Simulation interupted because of following error', 'red'))
        print(str(e))
        RUNNING = False
    finally:
        # close the simulation cleanly
        RUNNING = False
        print(colored('Closing simulator', 'green'))
        env.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
