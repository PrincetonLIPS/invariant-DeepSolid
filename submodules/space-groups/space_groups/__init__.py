"""
space_groups

A library for manipulating wallpaper and space groups in Python.
"""

__version__ = "0.1.0"
__author__  = 'Ryan P. Adams'
__credits__ = 'Princeton University'

from .group import SpaceGroup, PlaneGroup
from .fundamental import SpaceRegion, PlaneRegion
from .projective import up, dn
