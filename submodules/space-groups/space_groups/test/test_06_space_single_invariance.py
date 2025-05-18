import sys
sys.path.append("..")

import jax
import jax.numpy as jnp
import jax.random as jrnd
import unittest as ut
import numpy.testing as nptest

from nose2.tools import params

jax.config.update("jax_enable_x64", True)

from space_groups import SpaceGroup
from space_groups.invariance import construct_single

group_nums = range(1, 231)

class TestSpaceSingleInvariance(ut.TestCase):

  def setUp(self):
    self.rng = jrnd.PRNGKey(1)

  def rng_split(self):
    self.rng, rng = jrnd.split(self.rng)
    return rng

  def create_invariant_func(self, group, rng, num_feats=100, lengthscale=0.2):
    omega_rng = self.rng_split()
    phi_rng = self.rng_split()
    weights_rng = self.rng_split()

    omegas = jrnd.normal(omega_rng, shape=(6, num_feats)) / lengthscale
    phis = jrnd.uniform(phi_rng, shape=(num_feats,)) * 2 * jnp.pi
    weights = jrnd.normal(weights_rng, shape=(num_feats,)) / jnp.sqrt(num_feats)
    func = lambda x: jnp.cos(x @ omegas + phis) @ weights

    return construct_single(func, group)

  @params(*group_nums)
  def test_01_translation(self, group_num):
    '''Test for invariance to translations.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split())

    for ii in range(25):
      # Create a lattice of points consistent with the translations.
      point = jrnd.normal(self.rng_split(), shape=(3,))
      mesh = jnp.stack(
          jnp.meshgrid(*[jnp.arange(-2, 3) for _ in range(3)]),
          axis=-1,
        ).reshape(-1,3) + point[jnp.newaxis,:]
    
      # Validate that the invariant function is invariant to translations.
      vals = F(mesh)
      nptest.assert_allclose(vals, vals[0])

  @params(*group_nums)
  def test_02_random(self, group_num):
    '''Test for lack of invariance to random points.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split())

    for ii in range(25):
      # Create a random set of points to which we should not be invariant.
      points = jrnd.normal(self.rng_split(), shape=(100,3))
    
      # Validate that we are not equal across these points.
      vals = F(points)
      self.assertEqual(jnp.unique(vals).shape[0], vals.shape[0])

  @params(*group_nums)
  def test_03_actions(self, group_num):
    '''Test for invariance to non-translation group actions.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split())

    operations = jnp.array([op.to_doubles() for op in G.operations])

    for ii in range(25):

      # Create a random set of points.
      pts = jrnd.normal(self.rng_split(), shape=(100,3))

      # Apply the function to the mesh to get initial values.
      vals0 = F(pts)

      # Put points into homogeneous coordinates.
      pts = jnp.concatenate([pts, jnp.ones((pts.shape[0], 1))], axis=-1)

      for op in operations:
        # Apply the operation to the points.
        op_pts = pts @ op.T

        # Trim off the last dimension.
        op_pts = op_pts[...,:3]

        # Apply the invariant function.
        vals = F(op_pts)

        nptest.assert_allclose(vals0, vals)