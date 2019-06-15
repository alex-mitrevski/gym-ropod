from copy import deepcopy
from os.path import isfile
from typing import Tuple
from xml.etree import ElementTree

class ModelDescription(object):
    '''A description of a simulation model given in SDF format.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, name: str, sdf_path: str,
                 pose: Tuple[Tuple[float, float, float],
                             Tuple[float, float, float]]=None,
                 collision_size: Tuple[float, float, float]=None,
                 visual_size: Tuple[float, float, float]=None):
        '''Keyword arguments:

        name: str -- model name
        sdf_path: str -- path to the model SDF
        pose: Tuple -- a 3D pose representation in the form
                       (position, orientation), where position = (x, y, z)
                       and the orientation is represented using Euler angles, namely
                       orientation = (x, y, z)
        collision_size: Tuple -- size of the collision model in an (x, y, z) format
        visual_size: Tuple -- size of the visual model in an (x, y, z) format

        '''
        if not isfile(sdf_path):
            raise IOError('{0} is not a valid path'.format(sdf_path))
        self.description = self._load_model_description(sdf_path)

        self.name = name

        self.pose = None
        if pose is not None:
            self.pose = pose
        else:
            self.pose = ((0., 0., 0.), (0., 0., 0.))

        self.collision_size = None
        if collision_size is not None:
            self.collision_size = collision_size
        else:
            self.collision_size = (0., 0., 0.)

        self.visual_size = None
        if visual_size is not None:
            self.visual_size = visual_size
        else:
            self.visual_size = (0., 0., 0.)

    def as_string(self) -> str:
        '''Returns the model description in an XML string format.
        '''
        updated_description = self.set_model_parameters()
        return ElementTree.tostring(updated_description.getroot()).decode()

    def set_model_parameters(self) -> ElementTree.ElementTree:
        '''Simply returns self.description. Child classes needs to implement
        this method for properly setting the sizes of the collision and visual models.
        '''
        return self.description

    def _load_model_description(self, sdf_path: str) -> ElementTree.ElementTree:
        '''Returns an ElementTree with the contents of "sdf_path".

        Keyword arguments:
        sdf_path: str -- path to a model SDF file

        '''
        model_description = ElementTree.parse(sdf_path)
        return model_description


class PrimitiveModel(ModelDescription):
    '''A simulation model with a primitive geometry (box, sphere, cylinder).
    The model is assumed to contain a single link.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, name: str, sdf_path: str, model_type: str='box',
                 pose: Tuple[Tuple[float, float, float],
                             Tuple[float, float, float]]=None,
                 collision_size: Tuple[float, float, float]=None,
                 visual_size: Tuple[float, float, float]=None):
        '''Keyword arguments:

        name: str -- model name
        sdf_path: str -- path to the model SDF
        model_type: str -- geometry type (default "box")
        pose: Tuple -- a 3D pose representation in the form
                       (position, orientation), where position = (x, y, z)
                       and the orientation is represented using Euler angles, namely
                       orientation = (x, y, z)
        collision_size: Tuple -- size of the collision model in an (x, y, z) format
        visual_size: Tuple -- size of the visual model in an (x, y, z) format

        '''
        super(PrimitiveModel, self).__init__(name, sdf_path, pose, collision_size, visual_size)
        self.type = model_type

    def set_model_parameters(self) -> ElementTree.ElementTree:
        '''Returns a copy of self.description in which the sizes of the collisions
        and visual models have been updated with the values of self.collision_size
        and self.visual_size respectively.
        '''
        updated_description = deepcopy(self.description)

        # we look for the link element of the model
        model_element = updated_description.find('model')
        link_element = model_element.find('link')

        # we set the pose of the object
        pose_element = model_element.find('pose')
        pose_text = ' '.join([str(x) for x in self.pose[0]]) + ' ' + \
                    ' '.join([str(x) for x in self.pose[1]])
        pose_element.text = pose_text.strip()

        # we set the size of the collision model
        collision_element = link_element.find('collision')
        geometry_element = collision_element.find('geometry')
        collision_type_element = geometry_element.find(self.type)
        collision_size_element = collision_type_element.find('size')
        collision_size_element.text = ' '.join([str(x) for x in self.collision_size]).strip()

        # we set the size of the visual model
        visual_element = link_element.find('visual')
        geometry_element = visual_element.find('geometry')
        visual_type_element = geometry_element.find(self.type)
        visual_size_element = visual_type_element.find('size')
        visual_size_element.text = ' '.join([str(x) for x in self.visual_size]).strip()

        return updated_description
