from typing import Tuple

import os
import numpy as np
from gym import spaces
import rospy
from geometry_msgs.msg import Twist

from gym_ropod.envs.ropod_env import RopodEnv
from gym_ropod.utils.model import PrimitiveModel
from gym_ropod.utils.environment import EnvironmentDescription

class RopodNavActions(object):
    '''Defines the following navigation action mappings:
    action_num_to_str: Dict[int, str] -- maps integers describing actions
                                         (belonging to the action space
                                          gym.spaces.Discrete(5) to descriptive
                                          action names)
    action_to_vel: Dict[str, List[float]] -- maps action names to 2D velocity commands
                                             of the form [x, y, theta]

    @author Alex mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    action_num_to_str = {
        0: 'straight',
        1: 'left',
        2: 'right',
        3: 'left_turn',
        4: 'right_turn'
    }

    action_to_vel = {
        'straight': [0.1, 0.0, 0.0],
        'left': [0.0, 0.1, 0.0],
        'right': [0.0, -0.1, 0.0],
        'left_turn': [0.1, 0.0, 0.1],
        'right_turn': [0.1, 0.0, -0.1]
    }


class RopodNavEnvConfig(object):
    '''Defines the following environment mappings:
    env_to_config: Dict[string, EnvironmentDescription] -- maps environment names
                                                           to EnvironmentDescription objects
                                                           that configure the environments

    @author Alex mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    env_to_config = {
        'square': EnvironmentDescription(((-5., 5.), (-5., 5.)))
    }


class RopodNavDiscreteEnv(RopodEnv):
    '''A navigation environment for a ROPOD robot with a discrete action space.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, launch_file_path: str,
                 cmd_vel_topic: str='/ropod/cmd_vel',
                 env_type: str='square',
                 number_of_obstacles: int=0):
        '''Throws an AssertionError if "env_name" is not in RopodNavEnvConfig.env_to_config or
        the environment variable "ROPOD_GYM_MODEL_PATH" is not set.

        Keyword arguments:

        launch_file_path: str -- absolute path to a launch file that starts the ROPOD simulation
        cmd_vel_topic: str -- name of a ROS topic for base velocity commands
                              (default "/ropod/cmd_vel")
        env_type: str -- type of an environment (default "square")
        number_of_obstacles: int -- number of obstacles to add to the environment (default 0)

        '''
        super(RopodNavDiscreteEnv, self).__init__(launch_file_path)

        if env_type not in RopodNavEnvConfig.env_to_config:
            raise AssertionError('Unknown environment "{0}" specified'.format(env_type))

        if 'ROPOD_GYM_MODEL_PATH' not in os.environ:
            raise AssertionError('The ROPOD_GYM_MODEL_PATH environment variable is not set')

        self.model_path = os.environ['ROPOD_GYM_MODEL_PATH']
        self.env_config = RopodNavEnvConfig.env_to_config[env_type]
        self.number_of_obstacles = number_of_obstacles

        self.vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.vel_msg = Twist()

        self.action_space = spaces.Discrete(5)

    def step(self, action: int):
        '''Publishes a velocity command message based on the given action.

        Keyword arguments:
        action: int -- a navigation action to execute

        '''
        vels = RopodNavActions.action_to_vel[RopodNavActions.action_num_to_str[action]]

        self.vel_msg.linear.x = vels[0]
        self.vel_msg.linear.y = vels[1]
        self.vel_msg.angular.z = vels[2]

        self.vel_pub.publish(self.vel_msg)

    def reset(self):
        '''Resets the simulation environment.
        '''
        super().reset()

        for i in range(self.number_of_obstacles):
            pose, collision_size, visual_size = self.sample_model_parameters()
            model_name = 'box_' + str(i+1)
            model = PrimitiveModel(name=model_name,
                                   sdf_path=os.path.join(self.model_path, 'box.sdf'),
                                   pose=pose, collision_size=collision_size,
                                   visual_size=visual_size)
            self.insert_model(model)

    def sample_model_parameters(self) -> Tuple[Tuple, Tuple, Tuple]:
        '''Generates a random pose as well as collision and visual sizes
        for a dynamic model. The parameters are generated as follows:
        * for the pose, only the x and y positions and the z orientation are set;
          the position are sampled from the environment boundaries specified in
          self.env_config, while the orientation is sampled between -pi and pi radians
        * the collision sizes are sampled between 0.2 and 1.0 in all three directions
        * the visual size is the same as the collision size
        '''
        position_x = np.random.uniform(self.env_config.boundaries[0][0],
                                       self.env_config.boundaries[0][1])
        position_y = np.random.uniform(self.env_config.boundaries[1][0],
                                       self.env_config.boundaries[1][1])
        orientation_z = np.random.uniform(-np.pi, np.pi)
        pose = ((position_x, position_y, 0.), (0., 0., orientation_z))

        collision_size_x = np.random.uniform(0.2, 1.0)
        collision_size_y = np.random.uniform(0.2, 1.0)
        collision_size_z = np.random.uniform(0.2, 1.0)
        collision_size = (collision_size_x, collision_size_y, collision_size_z)
        visual_size = collision_size

        return (pose, visual_size, collision_size)
