from typing import Tuple

class EnvironmentDescription(object):
    '''Defines configuration parameters for an environment.

    @author Alex Mitrevski
    @contact aleksandar.mitrevski@h-brs.de

    '''
    def __init__(self, boundaries: Tuple[Tuple[float, float],
                                         Tuple[float, float]]):
        '''Keyword arguments:

        boundaries: Tuple -- environment coordinate boundaries in the format
                             ((min_x, max_x), (min_y, max_y))

        '''

        self.boundaries = boundaries
