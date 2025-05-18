import logging

from .symmetry import SymmetryGroup
from ..fundamental import SpaceRegion

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class SpaceGroup(SymmetryGroup):
  ''' Class containing basic information about space groups.

  Attributes:
  -----------

  number : int
    The number of the space group, taking values between 1 and 230.

  name : str
    The Hermann-Maguin notation for the space group, e.g., 'P 4 b m'.

  is_symmorphic : bool
    The space group is a semi-direct product between a point group and a
    translation group.

  is_entiantiomorphic : bool
    Is one of a pair of space groups with handedness.

  crystal_system : str
    One of 'Triclinic', 'Monoclinic', 'Orthorhombic', 'Tetragonal', 'Trigonal',
    'Hexagonal', or 'Cubic'.

  operations : List[RationalArray]
    A list of 4x4 RationalArrays, each of which corresponds to one of the group
    operations after quotienting out the translation subgroup.  These are in
    homogenous coordinates so they potentially perform both a point group
    operation and a translation.

  asu : SpaceRegion
    The asymmetric unit (fundamental region) for the space group, as specified
    in cctbx and the International Tables for Crystallography.  Note that this
    is not a unique region.

  wyckoff_positions : List[Dict]
    Each dictionary in the list contains the following keys:
      'op': 4x4 RationalArray
      'letter': Character identifying the site
      'multiplicity': Multiplicity of the site

  basic_basis : RationalArray
    A 3x3 RationalArray that represents a simple basis consistent with the
    necessary Bravais lattice for the group.

  '''

  RegionClass = SpaceRegion

  def __init__(self, group_idx: int, **kwargs):
    super().__init__(group_idx=group_idx, dims=3, **kwargs)
    self.dims = 3

  @property
  def _filename(self):
    return 'space_%03d.json' % (self.number)
