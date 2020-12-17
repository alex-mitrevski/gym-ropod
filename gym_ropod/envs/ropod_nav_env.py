from typing import Tuple, Sequence

import os
import pickle
import numpy as np
from gym import spaces

from gym_ropod.envs.ropod_env import RopodEnv, RopodEnvConfig
from gym_ropod.utils.model import PrimitiveModel
from gym_ropod.utils.geometry import GeometryUtils

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
        5: 'backward',
        6: 'do_nothing'
    }

    action_to_vel = {
        'forward': [0.8, 0.0, 0.0],
        'left': [0.0, 0.8, 0.0],
        'right': [0.0, -0.8, 0.0],
        'left_turn': [0.8, 0.0, 0.5],
        'right_turn': [0.8, 0.0, -0.5],
        'backward': [-0.8, 0.0, 0.0],
        'do_nothing': [0.0, 0.0, 0.0]
    }


class RopodNavActionsGP(object):
    '''Defines the following navigation action mapping for the learned GP Aggregate
    Primitive Model:
    action_num_to_str: Dict[int, str] -- maps integers describing actions
                                         (belonging to the action space
                                          gym.spaces.Discrete(5) to descriptive
                                          action names)
    '''
    action_num_to_str = {
        0: 'forward',
        1: 'left_rot_turn',
        2: 'right_rot_turn',
        3: 'left_arc_turn',
        4: 'right_arc_turn'
    }

    def action_to_vel(action_str, gp_model, segment_length=5, sum_over_steps=True):
        '''Samples a velocity command by sampling the GP model according to the given action.
        Returns:
        * the sampled action (an array of size (3,))

        Keyword arguments:
        action_str: str -- the name of the desired action
        gp_model: dict -- a dict of dicts containing the learned GP models
        segment_length: int -- the length (in timesteps) of the desired action sample
        sum_over_steps: bool -- if True, the vel commands sampled from the model are summed
                                across time (if segment_length > 1)
        '''
        x_space = np.arange(segment_length)[np.newaxis].T

        sample_x = gp_model[action_str]['x'].sample_y(x_space, random_state=None).squeeze()
        sample_y = gp_model[action_str]['y'].sample_y(x_space, random_state=None).squeeze()
        sample_theta = gp_model[action_str]['theta'].sample_y(x_space, random_state=None).squeeze()

        sampled_action = np.vstack((sample_x - sample_x[0],
                                    sample_y - sample_y[0],
                                    sample_theta - sample_theta[0])).T

        if sum_over_steps:
            sampled_action = sampled_action.sum(axis=0) / segment_length
        else:
            sampled_action = sampled_action / segment_length

        return sampled_action


class RopodNavDiscreteEnv(RopodEnv):
    '''A navigation environment for a ROPOD robot with a discrete action space.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, launch_file_path: str,
                 env_type: str='square',
                 number_of_obstacles: int=0,
                 use_gp_primitives: bool=True):
        '''Throws an AssertionError if "env_name" is not in RopodEnvConfig.env_to_config or
        the environment variable "ROPOD_GYM_MODEL_PATH" is not set.

        Keyword arguments:

        launch_file_path: str -- absolute path to a launch file that starts the ROPOD simulation
        cmd_vel_topic: str -- name of a ROS topic for base velocity commands
                              (default "/ropod/cmd_vel")
        env_type: str -- type of an environment (default "square")
        number_of_obstacles: int -- number of obstacles to add to the environment (default 0)

        '''
        super(RopodNavDiscreteEnv, self).__init__(launch_file_path)

        if env_type not in RopodEnvConfig.env_to_config:
            raise AssertionError('Unknown environment "{0}" specified'.format(env_type))

        if 'ROPOD_GYM_MODEL_PATH' not in os.environ:
            raise AssertionError('The ROPOD_GYM_MODEL_PATH environment variable is not set')

        self.model_path = os.environ['ROPOD_GYM_MODEL_PATH']
        self.env_config = RopodEnvConfig.env_to_config[env_type]
        self.number_of_obstacles = number_of_obstacles
        self.use_gp_primitives = use_gp_primitives

        self.action_space = spaces.Discrete(len(RopodNavActions.action_num_to_str))
        self.observation_space = spaces.Box(0., 5., (503,))

        self.collision_punishment = -1000.
        self.direction_change_punishment = -10.
        self.__inf = float('inf')

        self.goal_pose = None
        self.previous_action = None

        if self.use_gp_primitives:
            gp_model_filepath = '/home/lucy/workspace/ropod-rl/data/learned_gp_aggregate_primitive_model_1.pkl'
            self.primitive_gp_aggregate_model = self.__load_gp_model(gp_model_filepath)


    def step(self, action: int) -> Tuple[Tuple[float, float, float],
                                         Sequence[float], float, bool]:
        '''Publishes a velocity command message based on the given action.
        Returns:
        * a list in which
            * the first three elements represent the current goal the robot is pursuing
              (pose in the form (x, y, theta))
            * the subsequent elements represent the current laser scan measurements
        * obtained reward after performing the action
        * an indicator about whether the episode is done
        * an info dictionary containing a single key - "goal" -
          with the goal pose in the format (x, y, theta) as its value

        Keyword arguments:
        action: int -- a navigation action to execute

        '''
        # applying the action
        if self.use_gp_primitives:
            vels = RopodNavActionsGP.action_to_vel(RopodNavActionsGP.action_num_to_str[action],
                                                   self.primitive_gp_aggregate_model)
        else:
            vels = RopodNavActions.action_to_vel[RopodNavActions.action_num_to_str[action]]


        self.vel_msg.linear.x = vels[0]
        self.vel_msg.linear.y = vels[1]
        self.vel_msg.angular.z = vels[2]
        self.vel_pub.publish(self.vel_msg)

        # preparing the result
        reward = self.get_reward(action)
        observation = [x if x != self.__inf else self.laser_scan_msg.range_max
                       for x in self.laser_scan_msg.ranges]
        done = self.robot_under_collision or GeometryUtils.poses_equal(self.robot_pose,
                                                                       self.goal_pose)

        self.previous_action = action

        return (list(self.goal_pose) + observation, reward, done, {'goal': self.goal_pose})

    def get_reward(self, action: int) -> float:
        '''Calculates the reward obtained by applying the given action
        using the following equation:

        R_t = \frac{1}{d} + c_1\mathbf{1}_{c_t=1} + c_2\mathbf{a_{t-1} \neq a_t}

        where
        * d is the distance from the robot to the goal
        * c_t indicates whether the robot has collided
        * a_t is the action at time t
        * c_1 is the value of self.collision_punishment
        * c_2 is the vlaue of self.direction_change_punishment

        Keyword arguments:
        action: int -- an executed action

        '''
        goal_dist = GeometryUtils.distance(self.robot_pose, self.goal_pose)
        collision = 1 if self.robot_under_collision else 0
        direction_change = 1 if action != self.previous_action else 0
        reward = 1. / goal_dist + \
                 collision * self.collision_punishment + \
                 direction_change * self.direction_change_punishment
        return reward

    def reset(self):
        '''Resets the simulation environment. The first three elements of the
        returned observation represent the current goal the robot is pursuing
        (pose in the form (x, y, theta)); the subsequent elements represent
        the current laser measurements.
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

        # we generate a goal pose for the robot
        self.goal_pose = self.generate_goal_pose()

        # preparing the result
        observation = [x if x != self.__inf else self.laser_scan_msg.range_max
                       for x in self.laser_scan_msg.ranges]
        return list(self.goal_pose) + observation

    def generate_goal_pose(self) -> Tuple[float, float, float]:
        '''Randomly generates a goal pose in the environment, ensuring that
        the pose does not overlap any of the existing objects.
        '''
        goal_pose_found = False
        pose = None
        while not goal_pose_found:
            position_x = np.random.uniform(self.env_config.boundaries[0][0],
                                           self.env_config.boundaries[0][1])
            position_y = np.random.uniform(self.env_config.boundaries[1][0],
                                           self.env_config.boundaries[1][1])
            orientation_z = np.random.uniform(-np.pi, np.pi)

            pose = (position_x, position_y, orientation_z)
            if not self.__pose_overlapping_models(pose):
                goal_pose_found = True
        return pose

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

    def __pose_overlapping_models(self, pose: Tuple[float, float, float]):
        '''Returns True if the given pose overlaps with any of the existing
        objects in the environment; returns False otherwise.

        Keyword arguments:
        pose: Tuple[float, float, float]: a 2D pose in the format (x, y, theta)

        '''
        for model in self.models:
            if GeometryUtils.pose_inside_model(pose, model):
                return True
        return False

    def __load_gp_model(self, filepath):
        '''Loads the learned GP aggregate primitive model from the designated
        pickle file.
        Returns:
        * the dict of dicts containing the learned GP models

        Keyword arguments:
        filepath: str -- the path to the pkl file
        '''
        with open(filepath, 'rb') as f:
            return pickle.load(f)
