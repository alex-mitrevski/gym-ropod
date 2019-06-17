from typing import Tuple

import os
import numpy as np
from gym import spaces

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
        0: 'forward',
        1: 'left',
        2: 'right',
        3: 'left_turn',
        4: 'right_turn',
        5: 'backward'
    }

    action_to_vel = {
        'forward': [0.1, 0.0, 0.0],
        'left': [0.0, 0.1, 0.0],
        'right': [0.0, -0.1, 0.0],
        'left_turn': [0.1, 0.0, 0.1],
        'right_turn': [0.1, 0.0, -0.1],
        'backward': [-0.1, 0.0, 0.0]
    }


class RopodNavEnvConfig(object):
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


class RopodNavDiscreteEnv(RopodEnv):
    '''A navigation environment for a ROPOD robot with a discrete action space.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, launch_file_path: str,
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

        self.action_space = spaces.Discrete(len(RopodNavActions.action_num_to_str))

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

        # we add the static environment models
        for model in self.env_config.models:
            self.insert_env_model(model)

        # we add obstacles to the environment
        for i in range(self.number_of_obstacles):
            pose, collision_size, visual_size = self.sample_model_parameters()
            model_name = 'box_' + str(i+1)
            model = PrimitiveModel(name=model_name,
                                   sdf_path=os.path.join(self.model_path, 'models/box.sdf'),
                                   model_type='box', pose=pose,
                                   collision_size=collision_size,
                                   visual_size=visual_size)
            self.insert_dynamic_model(model)

    def sample_model_parameters(self) -> Tuple[Tuple, Tuple, Tuple]:
        '''Generates a random pose as well as collision and visual sizes
        for a dynamic model. The parameters are generated as follows:
        * for the pose, only the position and z-orientation are set;
          the x and y positions are sampled from the environment boundaries specified in
          self.env_config, the orientation is sampled between -pi and pi radians, and
          the z position is set to half the z collision size in order for the model
          to be on top of the ground
        * the collision sizes are sampled between 0.2 and 1.0 in all three directions
        * the visual size is the same as the collision size
        '''
        collision_size_x = np.random.uniform(0.2, 1.0)
        collision_size_y = np.random.uniform(0.2, 1.0)
        collision_size_z = np.random.uniform(0.2, 1.0)
        collision_size = (collision_size_x, collision_size_y, collision_size_z)
        visual_size = collision_size

        position_x = np.random.uniform(self.env_config.boundaries[0][0],
                                       self.env_config.boundaries[0][1])
        position_y = np.random.uniform(self.env_config.boundaries[1][0],
                                       self.env_config.boundaries[1][1])
        position_z = collision_size_z / 2.
        orientation_z = np.random.uniform(-np.pi, np.pi)
        pose = ((position_x, position_y, position_z), (0., 0., orientation_z))

        return (pose, visual_size, collision_size)
