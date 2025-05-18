import jsonpickle
import logging
import pathlib
import re
import numpy as np

from .. import rational_numpy as rnp
from .symmetry import SymmetryGroup
from ..fundamental import PlaneRegion
from ..projective import up, dn

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class PlaneGroup(SymmetryGroup):

  ''' Class containing basic information about plane groups.

  Attributes:
  -----------

  number : int
    The number of the space group, taking values between 1 and 17.

  name : str
    The Hermann-Maguin notation for the plane group, e.g., 'pg'.

  crystal_system : str
    One of 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal',
    'Hexagonal', or 'Cubic'.

  operations : List[RationalArray]
    A list of 3x3 RationalArrays, each of which corresponds to one of the group
    operations after quotienting out the translation subgroup.  These are in
    homogenous coordinates so they potentially perform both a point group
    operation and a translation.

  asu : PlaneRegion
    The asymmetric unit (fundamental region) for the space group, as specified
    in cctbx and the International Tables for Crystallography.  Note that this
    is not a unique region.

  basic_basis : RationalArray
    A 2x2 RationalArray that represents a simple basis consistent with the
    necessary Bravais lattice for the group.

  '''

  # Each plane group corresponds to a projection of a space group.
  # We use this map to identify the right one and load its properties.
  #space_group_map = {
  #  1: ('p1', 1),
  #  2: ('p2', 3),
  #  3: ('pm', 6),
  #  4: ('pg', 7),
  #  5: ('cm', 8),
  #  6: ('p2mm', 25),
  #  7: ('p2mg', 28),
  #  8: ('p2gg', 32),
  #  9: ('c2mm', 35),
  #  10: ('p4', 75),
  #  11: ('p4mm', 99),
  #  12: ('p4gm', 100),
  #  13: ('p3', 143),
  #  14: ('p3m1', 156),
  #  15: ('p31m', 157),
  #  16: ('p6', 168),
  #  17: ('p6mm', 183),
  #}

  space_group_map = {
    1: ('p1', 'P 1'),
    2: ('p2', 'P 2'),
    3: ('pm', 'P -2x'),
    4: ('pg', 'P -2xb'),
    5: ('cm', 'C -2x'),
    6: ('p2mm', 'P 2 -2'),
    7: ('p2mg', 'P 2 -2a'),
    8: ('p2gg', 'P 2 -2ab'),
    9: ('c2mm', 'C 2 -2'),
    10: ('p4', 'P 4'),
    11: ('p4mm', 'P 4 -2'),
    12: ('p4gm', 'P 4 -2ab'),
    13: ('p3', 'P 3'),
    14: ('p3m1', 'P 3 -2"'),
    15: ('p31m', 'P 3 -2'),
    16: ('p6', 'P 6'),
    17: ('p6mm', 'P 6 -2'),
    }


  RegionClass = PlaneRegion

  def __init__(self, group_idx: int, **kwargs) -> None:
    super().__init__(group_idx=group_idx, dims=2, **kwargs)
    self.dims = 2

  @property
  def _filename(self):
    return 'plane_%02d.json' % (self.number)

  def orbit_points(self, pts, txty=None):

    # Apply all of the unit cell symmetries.
    pts = up(pts)
    ops = np.array([op.to_doubles() for op in self.operations])
    unit_orbits = np.tensordot(pts, ops, axes=(1, 2))
    unit_orbits = np.swapaxes(unit_orbits, 0, 1).reshape(-1,3)

    # Apply all of the translations.
    if txty is None:
      grid = [0, -2, -1, 1, 2]
      txty = np.stack(np.meshgrid(grid, grid), -1).reshape(-1,2)
    orbits = dn(unit_orbits)[:,np.newaxis,:] + txty[np.newaxis,:,:]
    orbits = np.reshape(np.swapaxes(orbits, 0,1), (-1,2))

    return orbits