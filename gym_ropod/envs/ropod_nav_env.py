from gym import spaces
import rospy
from geometry_msgs.msg import Twist

from gym_ropod.envs.ropod_env import RopodEnv

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


class RopodNavDiscreteEnv(RopodEnv):
    '''A navigation environment for a ROPOD robot with a discrete action space.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    Constructor arguments:
    launch_file_path: str -- absolute path to a launch file that starts the ROPOD simulation
    cmd_vel_topic: str -- name of a ROS topic for base velocity commands

    '''
    def __init__(self, launch_file_path: str,
                 cmd_vel_topic: str='/ropod/cmd_vel'):
        super(RopodNavDiscreteEnv, self).__init__(launch_file_path)

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
        self.reset_sim_proxy()
