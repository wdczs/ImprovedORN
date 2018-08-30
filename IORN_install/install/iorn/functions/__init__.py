from .RIE import ORAlign1d
from .RIE import ORAlign2d
from .ARF import MappingRotate

def oralign1d(input, nOrientation=4, return_direction=False):
  return ORAlign1d(nOrientation, return_direction)(input)

def oralign2d(input, nOrientation=4, return_direction=False):
  return ORAlign2d(nOrientation, return_direction)(input)

def mapping_rotate(input, indices):
  return MappingRotate(indices)(input)
  
__all__ = ["oralign1d", "oralign2d", "mapping_rotate"]