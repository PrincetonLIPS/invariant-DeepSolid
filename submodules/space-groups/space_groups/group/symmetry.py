import jsonpickle
import logging
import numpy as np
import pathlib
import re

from .. import rational_numpy as rnp

DEFAULT_DATA_DIR = pathlib.Path(__file__).parent.parent / 'data'

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class SymmetryGroup:
  basic_unit_cells = {
    'Triclinic':    (1, 1, 1, 90, 90, 90),
    'Monoclinic':   (1, 1, 1, 90, 90, 90),
    'Orthorhombic': (1, 1, 1, 90, 90, 90),
    'Tetragonal':   (1, 1, 1, 90, 90, 90),
    'Trigonal':     (1, 1, 1, 90, 90, 120),
    'Hexagonal':    (1, 1, 1, 90, 90, 120),
    'Cubic':        (1, 1, 1, 90, 90, 90),
  }

  def __init__(self, group_idx, dims, data_dir=DEFAULT_DATA_DIR):
    self.number = group_idx

    # Init from JSON.
    filename = pathlib.Path(data_dir) / self._filename
    log.debug('Loading from file %s.' % (filename))
    if not pathlib.Path(filename).exists():
      raise FileNotFoundError('Could not find file %s.' % (filename))
    with open(filename, 'r') as fh:
      json_string = fh.read()
      obj = jsonpickle.decode(json_string)
      self.__dict__.update(obj.__dict__)
