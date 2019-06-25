from abc import abstractmethod

import os
import subprocess
import time
from termcolor import colored
import transforms3d as tf

import gym

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import std_srvs.srv as std_srvs

from gazebo_msgs.msg import ContactsState, ModelStates
import gazebo_msgs.srv as gazebo_srvs

from gym_ropod.utils.model import ModelDescription
from gym_ropod.utils.environment import EnvironmentDescription

class RopodEnvConfig(object):
    '''Defines the following environment mappings:
    env_to_config: Dict[string, EnvironmentDescription] -- maps environment names
                                                           to EnvironmentDescription objects
                                                           that configure the environments

    @author Alex mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    world_path = os.environ['ROPOD_GYM_MODEL_PATH']
    env_to_config = {
        'square': EnvironmentDescription(os.path.join(world_path, 'worlds/square.world'),
                                         ((-20, 0.), (0., 20.)))
    }


class RopodEnv(gym.Env):
    '''An abstract base class for ROPOD environments. Builds upon
    https://github.com/ascane/gym-gazebo-hsr/blob/master/gym_gazebo_hsr/envs/gazebo_env.py

    '''
    def __init__(self, launch_file_path: str,
                 roscore_port: str='11311',
                 reset_sim_srv_name: str='/gazebo/reset_world',
                 spawn_model_srv_name: str='/gazebo/spawn_sdf_model',
                 delete_model_srv_name: str='/gazebo/delete_model',
                 model_state_topic: str='/gazebo/model_states',
                 cmd_vel_topic: str='/ropod/cmd_vel',
                 laser_topic: str='/ropod/laser/scan',
                 bumper_topic: str='/ropod/bumper',
                 output_file: str=None):
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
        model_state_topic: str -- name of a topic for getting the state of the simulation models
                                  (default "/gazebo/model_states")
        cmd_vel_topic: str -- name of a topic at which velocity commands are published
                              (default "/ropod/cmd_vel")
        laser_topic: str -- name of a topic at which laser scans are published
                            (default "/ropod/laser/scan")
        bumper_topic: str -- name of a topic at which collision status can be obtained
                             (default "/ropod/bumper")
        output_file: str -- file path where the output of the child process should be stored
                             (default "None")

        '''
        if not os.path.exists(launch_file_path):
            raise IOError('{0} is not a valid launch file path'.format(launch_file_path))

        if not output_file:
            output_file = '/tmp/gym_ropod_' + str(time.time()).replace('.','_')

        self.roscore_process = None
        self.sim_process = None
        self.reset_sim_proxy = None
        self.spawn_model_proxy = None
        self.delete_model_proxy = None
        self.sim_vis_process = None

        self.environment_model_names = []
        self.dynamic_model_names = []
        self.models = []

        self.output_file_obj = open(output_file, 'w')

        print(colored('[RopodEnv] Launching roscore...', 'green'))
        self.roscore_process = subprocess.Popen(['roscore', '-p', roscore_port], stdout=self.output_file_obj)
        time.sleep(1)
        print(colored('[RopodEnv] Roscore launched!', 'green'))

        print(colored('[RopodEnv] Launching simulator...', 'green'))
        self.sim_process = subprocess.Popen(['roslaunch', '-p', roscore_port,
                                             launch_file_path, 'gui:=false'],
                                             stdout=self.output_file_obj)
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

        self.vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.vel_msg = Twist()

        self.laser_scan_msg = None
        self.laser_scan_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_cb)

        self.robot_under_collision = False
        self.bumper_sub = rospy.Subscriber(bumper_topic, ContactsState, self.bumper_cb)

        self.robot_pose = None
        self.model_state_sub = rospy.Subscriber(model_state_topic, ModelStates, self.save_robot_pose)

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
        for model_name in self.environment_model_names:
            self.__delete_model(model_name)
            self.environment_model_names.remove(model_name)

        for model_name in self.dynamic_model_names:
            self.__delete_model(model_name)
            self.dynamic_model_names.remove(model_name)

        self.models = []
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
        self.output_file_obj.close()

    def laser_cb(self, msg: LaserScan) -> None:
        '''Saves "msg" in self.laser_scan_msg.
        '''
        self.laser_scan_msg = msg

    def bumper_cb(self, msg: ContactsState) -> None:
        '''Updates the value of self.robot_under_collision
        depending on whether "msg" includes information about
        contacts with the environment.
        '''
        if msg.states:
            self.robot_under_collision = True
        else:
            self.robot_under_collision = False

    def save_robot_pose(self, msg: ModelStates) -> None:
        '''Updates the value of self.robot_pose based on the pose
        of the "robot" model in the given message.
        '''
        for i, model_name in enumerate(msg.name):
            if model_name == 'ropod':
                position_x = msg.pose[i].position.x
                position_y = msg.pose[i].position.y

                quat_orientation = (msg.pose[i].orientation.w, msg.pose[i].orientation.x,
                                    msg.pose[i].orientation.y, msg.pose[i].orientation.z)
                euler_orientation = tf.euler.quat2euler(quat_orientation)
                orientation_z = euler_orientation[2]
                self.robot_pose = (position_x, position_y, orientation_z)
                return

    def insert_env_model(self, model: ModelDescription) -> None:
        '''Adds a static environment model to the simulation.
        Removes the model and adds it again if it already exists in the environment.

        Keyword arguments:
        model: ModelDescription -- model parameters

        '''
        if model.name in self.environment_model_names:
            print(colored('[RopodEnv] Removing existing model "{0}"'.format(model.name), 'yellow'))
            self.__delete_model(model.name)
            self.environment_model_names.remove(model.name)

        self.__insert_model(model)
        self.environment_model_names.append(model.name)
        self.models.append(model)

    def insert_dynamic_model(self, model: ModelDescription) -> None:
        '''Adds a dynamic model to the simulated environment.
        Removes the model and adds it again if it already exists in the environment.

        Keyword arguments:
        model: ModelDescription -- model parameters

        '''
        if model.name in self.dynamic_model_names:
            print(colored('[RopodEnv] Removing existing model "{0}"'.format(model.name), 'yellow'))
            self.__delete_model(model.name)
            self.dynamic_model_names.remove(model.name)

        self.__insert_model(model)
        self.dynamic_model_names.append(model.name)
        self.models.append(model)

    def __insert_model(self, model: ModelDescription) -> None:
        '''Adds a model to the simulated environment.

        Keyword arguments:
        model: ModelDescription -- model parameters

        '''
        model_request = gazebo_srvs.SpawnModelRequest()
        model_request.model_name = model.name
        model_request.model_xml = model.as_string()
        model_request.reference_frame = 'world'

        model_request.initial_pose.position.x = model.canonical_pose[0][0]
        model_request.initial_pose.position.y = model.canonical_pose[0][1]
        model_request.initial_pose.position.z = model.canonical_pose[0][2]

        quat_orientation = tf.euler.euler2quat(*model.canonical_pose[1])
        model_request.initial_pose.orientation.w = quat_orientation[0]
        model_request.initial_pose.orientation.x = quat_orientation[1]
        model_request.initial_pose.orientation.y = quat_orientation[2]
        model_request.initial_pose.orientation.z = quat_orientation[3]

        print(colored('[RopodEnv] Inserting model "{0}"'.format(model.name), 'green'))
        response = self.spawn_model_proxy(model_request)
        if response.success:
            print(colored('[RopodEnv] Model "{0}" inserted'.format(model.name), 'green'))
        else:
            print(colored('[RopodEnv] Could not insert "{0}": {1}'.format(model.name,
                                                                          response.status_message), 'red'))

    def __delete_model(self, model_name: str) -> None:
        '''Removes a model from the simulation.

        Keyword arguments:
        model_name: str -- name of the model to remove

        '''
        delete_model_request = gazebo_srvs.DeleteModelRequest()
        delete_model_request.model_name = model_name

        print(colored('[RopodEnv] Deleting model "{0}"'.format(model_name), 'green'))
        response = self.delete_model_proxy(delete_model_request)
        if response.success:
            print(colored('[RopodEnv] Model "{0}" deleted'.format(model_name), 'green'))
        else:
            print(colored('[RopodEnv] Could not delete "{0}": {1}'.format(model_name,
                                                                          response.status_message), 'red'))

    def _close_sim_client(self):
        '''Stops the process running the simulation client.
        '''
        if self.sim_vis_process is not None and self.sim_vis_process.poll() is None:
            self.sim_vis_process.terminate()
            self.sim_vis_process.wait()
