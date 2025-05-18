
import sys
sys.path.append("..")

import unittest as ut

from nose2.tools import params

from space_groups import SpaceGroup

group_nums = range(1, 231)

class TestSpaceGroups(ut.TestCase):

  def setUp(self):
    pass

  @params(*group_nums)
  def test_01_basic(self, group_num):
    G = SpaceGroup(group_num)
    self.assertEqual(G.number, group_num)
    self.assertEqual(G.dims, 3)
    self.assertEqual(G.basic_basis.shape, (3, 3))
    
