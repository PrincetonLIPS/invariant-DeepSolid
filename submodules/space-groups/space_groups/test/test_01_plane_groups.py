
import sys
sys.path.append("..")

import unittest as ut

from nose2.tools import params

from space_groups import PlaneGroup

group_nums = range(1, 18)

class TestPlaneGroups(ut.TestCase):

  def setUp(self):
    pass

  @params(*group_nums)
  def test_01_basic(self, group_num):
    G = PlaneGroup(group_num)
    self.assertEqual(G.number, group_num)
    self.assertEqual(G.dims, 2)
    self.assertEqual(G.basic_basis.shape, (2, 2))
    