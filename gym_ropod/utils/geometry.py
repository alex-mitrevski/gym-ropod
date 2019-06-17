from typing import Tuple
import numpy as np
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from gym_ropod.utils.model import PrimitiveModel

class GeometryUtils(object):
    @staticmethod
    def distance(pos1: Tuple, pos2: Tuple):
        '''Calculates the distance between two positions.
        Raises an AssertionError if the positions are not of the same length.

        Keyword arguments:
        pos1: Tuple: n-dimensional position
        pos2: Tuple: n-dimensional position

        '''
        if len(pos1) != len(pos2):
            raise AssertionError('[GeometryUtils.distance] pos1 and pos2 need to have the same length')

        return np.linalg.norm([np.square(pos1[i] - pos2[i]) for i in list(range(len(pos1)))])

    @staticmethod
    def poses_equal(pose1: Tuple[float, float, float],
                    pose2: Tuple[float, float, float],
                    position_tolerance: float=0.1,
                    orientation_tolerance: float=0.1) -> bool:
        '''Returns True if the differences in position and orientation
        of the given poses are within predefined thresholds.

        Keyword arguments:
        pose1: Tuple[float, float, float] -- a 2D pose in the format (x, y, theta)
        pose2: Tuple[float, float, float] -- a 2D pose in the format (x, y, theta)
        position_tolerance: float -- difference threshold between two linear
                                     coordinates in meters (default 0.1)
        orientation_tolerance: float -- difference threshold between two orientations
                                        in radians (default 0.1)

        '''
        return abs(pose1[0] - pose2[0]) < position_tolerance and \
               abs(pose1[1] - pose2[1]) < position_tolerance and \
               abs(pose1[2] - pose2[2]) < orientation_tolerance

    @staticmethod
    def pose_inside_model(pose: Tuple[float, float, float],
                          model: PrimitiveModel) -> bool:
        '''Returns True if the given position is in the polygon obtained
        by projecting the given model onto the xy-plane and False otherwise.

        Keyword arguments:
        pose: Tuple[float, float, float] -- a 2D pose
        model: PrimitiveModel -- a model of an object in the environment

        '''
        point = Point(pose[0], pose[1])
        model_polygon = Polygon(((model.pose[0][0] - model.visual_size[0],
                                  model.pose[0][1] - model.visual_size[1]),
                                 (model.pose[0][0] - model.visual_size[0],
                                  model.pose[0][1] + model.visual_size[1]),
                                 (model.pose[0][0] + model.visual_size[0],
                                  model.pose[0][1] + model.visual_size[1]),
                                 (model.pose[0][0] + model.visual_size[0],
                                  model.pose[0][1] - model.visual_size[1])))
        return point.within(model_polygon)
