from typing import Tuple
from xml.etree import ElementTree

class SDFUtils(object):
    @staticmethod
    def load_description(sdf_path: str) -> ElementTree.Element:
        '''Returns an element with the contents of "sdf_path".

        Keyword arguments:
        sdf_path: str -- path to a model SDF file

        '''
        model_description = ElementTree.parse(sdf_path)
        return model_description.getroot()

    @staticmethod
    def get_pose(model_element: ElementTree.Element) -> Tuple[Tuple, Tuple]:
        '''Returns the pose of the given element.
        '''
        pose_element = model_element.find('pose')
        pose_items = [float(x) for x in pose_element.text.split(' ')]

        position = (pose_items[0], pose_items[1], pose_items[2])
        orientation = (pose_items[3], pose_items[4], pose_items[5])
        pose = (position, orientation)
        return pose

    @staticmethod
    def get_canonical_pose(model_element: ElementTree.Element) -> Tuple[Tuple, Tuple]:
        '''Returns the pose of the element's canonical link.
        '''
        link_element = model_element.find('link')
        pose_element = link_element.find('pose')
        pose_items = [float(x) for x in pose_element.text.split(' ')]

        position = (pose_items[0], pose_items[1], pose_items[2])
        orientation = (pose_items[3], pose_items[4], pose_items[5])
        pose = (position, orientation)
        return pose

    @staticmethod
    def get_collision_size(model_element: ElementTree.Element,
                           collision_model_type: str) -> Tuple[Tuple, Tuple]:
        '''Returns the size of the element's collision model.
        '''
        link_element = model_element.find('link')
        collision_element = link_element.find('collision')
        geometry_element = collision_element.find('geometry')
        collision_type_element = geometry_element.find(collision_model_type)
        collision_size_element = collision_type_element.find('size')
        collision_size = [float(x) for x in collision_size_element.text.split(' ')]
        return collision_size

    @staticmethod
    def get_visual_size(model_element: ElementTree.Element,
                        visual_model_type: str) -> Tuple[Tuple, Tuple]:
        '''Returns the element's visual model.
        '''
        link_element = model_element.find('link')
        visual_element = link_element.find('visual')
        geometry_element = visual_element.find('geometry')
        visual_type_element = geometry_element.find(visual_model_type)
        visual_size_element = visual_type_element.find('size')
        visual_size = [float(x) for x in visual_size_element.text.split(' ')]
        return visual_size
