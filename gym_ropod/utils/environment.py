from typing import Tuple, Sequence
from os.path import isfile
from xml.etree import ElementTree

from gym_ropod.utils.sdf import SDFUtils
from gym_ropod.utils.model import PrimitiveModel

class EnvironmentDescription(object):
    '''Represents the configuration of an environment including models
    for reconstructing the environment and environment boundaries.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, sdf_path: str, boundaries: Tuple[Tuple[float, float],
                                                        Tuple[float, float]]):
        '''Keyword arguments:

        sdf_path: str -- path to the environment SDF
        boundaries: Tuple -- environment coordinate boundaries in the format
                             ((min_x, max_x), (min_y, max_y))

        '''
        if not isfile(sdf_path):
            raise IOError('{0} is not a valid path'.format(sdf_path))
        self.description = SDFUtils.load_description(sdf_path)
        self.boundaries = boundaries

        model_elements = self.__extract_model_elements()
        self.models = self.__create_models(model_elements)

    def __create_models(self, model_elements: Sequence[ElementTree.Element]) -> Sequence[PrimitiveModel]:
        models = []
        for model_element in model_elements:
            name = model_element.attrib['name']
            pose = SDFUtils.get_pose(model_element)
            canonical_pose = SDFUtils.get_canonical_pose(model_element)
            collision_size = SDFUtils.get_collision_size(model_element, 'box')
            visual_size = SDFUtils.get_visual_size(model_element, 'box')

            model = PrimitiveModel(name=name, xml_element=model_element,
                                   model_type='box', pose=pose,
                                   canonical_pose=canonical_pose,
                                   collision_size=collision_size,
                                   visual_size=visual_size)
            models.append(model)
        return models

    def __extract_model_elements(self) -> Sequence[ElementTree.Element]:
        '''Returns a list of all "model" elements in self.description.
        The method assumes that the first element in the element tree
        is a "world" element and that the models are its direct children.
        '''
        world_element = self.description.find('world')
        model_elements = world_element.findall('model')
        return model_elements
