from abc import abstractmethod

import os
import subprocess
import time
from termcolor import colored

import gym

import rospy
import std_srvs.srv as std_srvs
import gazebo_msgs.srv as gazebo_srvs
import transforms3d as tf

from gym_ropod.models.model_description import ModelDescription

class RopodEnv(gym.Env):
    '''An abstract base class for ROPOD environments. Builds upon
    https://github.com/ascane/gym-gazebo-hsr/blob/master/gym_gazebo_hsr/envs/gazebo_env.py

    '''
    def __init__(self, launch_file_path: str,
                 roscore_port: str='11311',
                 reset_sim_srv_name: str='/gazebo/reset_world',
                 spawn_model_srv_name: str='/gazebo/spawn_sdf_model',
                 delete_model_srv_name: str='/gazebo/delete_model'):
        '''Raises an IOError if the specified launch file path does not exist.

        Keyword arguments:
        launch_file_path: str -- absolute path to a launch file that starts the ROPOD simulation
        roscore_port: str -- port on which the ROS master should be run (default "11311")
        reset_sim_srv_name: str -- name of a service that resets the simulated environment
                                   (default "/gazebo/reset_world")
        spawn_model_srv_name: str -- name of a service for adding models to the environment
                                     (default "/gazebo/spawn_sdf_model")
        delete_model_srv_name: str -- name of a service for deleting models from the environment
                                      (default "/gazebo/delete_model")

        '''
        if not os.path.exists(launch_file_path):
            raise IOError('{0} is not a valid launch file path'.format(launch_file_path))

        self.roscore_process = None
        self.sim_process = None
        self.reset_sim_proxy = None
        self.spawn_model_proxy = None
        self.delete_model_proxy = None
        self.sim_vis_process = None

        self.dynamic_model_names = []

        print(colored('[RopodEnv] Launching roscore...', 'green'))
        self.roscore_process = subprocess.Popen(['roscore', '-p', roscore_port])
        time.sleep(1)
        print(colored('[RopodEnv] Roscore launched!', 'green'))

        print(colored('[RopodEnv] Launching simulator...', 'green'))
        self.sim_process = subprocess.Popen(['roslaunch', '-p', roscore_port,
                                             launch_file_path, 'gui:=false'])
        print(colored('[RopodEnv] Simulator launched!', 'green'))

        print(colored('[RopodEnv] Waiting for service {0}'.format(reset_sim_srv_name), 'green'))
        rospy.wait_for_service(reset_sim_srv_name)
        self.reset_sim_proxy = rospy.ServiceProxy(reset_sim_srv_name, std_srvs.Empty)
        print(colored('[RopodEnv] Service {0} is up'.format(reset_sim_srv_name), 'green'))

        print(colored('[RopodEnv] Waiting for service {0}'.format(spawn_model_srv_name), 'green'))
        rospy.wait_for_service(spawn_model_srv_name)
        self.spawn_model_proxy = rospy.ServiceProxy(spawn_model_srv_name, gazebo_srvs.SpawnModel)
        print(colored('[RopodEnv] Service {0} is up'.format(spawn_model_srv_name), 'green'))

        print(colored('[RopodEnv] Waiting for service {0}'.format(delete_model_srv_name), 'green'))
        rospy.wait_for_service(delete_model_srv_name)
        self.delete_model_proxy = rospy.ServiceProxy(delete_model_srv_name, gazebo_srvs.DeleteModel)
        print(colored('[RopodEnv] Service {0} is up'.format(delete_model_srv_name), 'green'))

        print(colored('[RopodEnv] Initialising ROS node', 'green'))
        rospy.init_node('gym')
        print(colored('[RopodEnv] ROS node initialised', 'green'))

    @abstractmethod
    def step(self, action: int):
        '''Runs a single step through the simulation.

        Keyword arguments:
        action: int -- an action to execute

        '''
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        '''Resets the simulation environment by removing all models
        that were dynamically added and resetting the state of all
        initial models.
        '''
        for model_name in self.dynamic_model_names:
            self.delete_model(model_name)
        self.reset_sim_proxy()

    def render(self, mode: str='human') -> None:
        '''Displays the current environment. Opens up a simulation process
        if the environment is being rendered for the first time.

        Keyword arguments:
        mode: str -- rendering mode (default "human")

        '''
        if self.sim_vis_process is None or self.sim_vis_process.poll() is not None:
            self.sim_vis_process = subprocess.Popen('gzclient')

    def close(self) -> None:
        '''Closes the simulation client and terminates the simulation and roscore processes.
        '''
        self._close_sim_client()
        self.sim_process.terminate()
        self.roscore_process.terminate()
        self.sim_process.wait()
        self.roscore_process.wait()

    def insert_model(self, model: ModelDescription) -> None:
        '''Adds a model to the simulated environment.

        Keyword arguments:
        model: ModelDescription -- model parameters

        '''
        if model.name in self.dynamic_model_names:
            print(colored('[RopodEnv] Removing existing model "{0}"'.format(model.name), 'yellow'))
            self.delete_model(model.name)

        model_request = gazebo_srvs.SpawnModelRequest()
        model_request.model_name = model.name
        model_request.model_xml = model.as_string()

        model_request.initial_pose.position.x = model.pose[0][0]
        model_request.initial_pose.position.y = model.pose[0][1]
        model_request.initial_pose.position.z = model.pose[0][2]

        quat_orientation = tf.euler.euler2quat(*model.pose[1])
        model_request.initial_pose.orientation.w = quat_orientation[0]
        model_request.initial_pose.orientation.x = quat_orientation[1]
        model_request.initial_pose.orientation.y = quat_orientation[2]
        model_request.initial_pose.orientation.z = quat_orientation[3]

        print(colored('[RopodEnv] Inserting model "{0}"'.format(model.name), 'green'))
        self.spawn_model_proxy(model_request)
        print(colored('[RopodEnv] Model "{0}" inserted'.format(model.name), 'green'))

        self.dynamic_model_names.append(model.name)

    def delete_model(self, model_name: str) -> None:
        '''Removes a model from the simulation.

        Keyword arguments:
        model_name: str -- name of the model to remove

        '''
        if model_name not in self.dynamic_model_names:
            print(colored('[RopodEnv] Model "{0}" does not exist, so cannot remove it', 'yellow'))
            return

        delete_model_request = gazebo_srvs.DeleteModelRequest()
        delete_model_request.model_name = model_name

        print(colored('[RopodEnv] Deleting model "{0}"'.format(model_name), 'green'))
        self.delete_model_proxy(delete_model_request)
        print(colored('[RopodEnv] Model "{0}" deleted'.format(model_name), 'green'))

        self.dynamic_model_names.remove(model_name)

    def _close_sim_client(self):
        '''Stops the process running the simulation client.
        '''
        if self.sim_vis_process is not None and self.sim_vis_process.poll() is None:
            self.sim_vis_process.terminate()
            self.sim_vis_process.wait()
