import sys
sys.path.append("..")

import jax
import jax.numpy as jnp
import jax.random as jrnd
import unittest as ut
import numpy.testing as nptest

from itertools import product
from nose2.tools import params

jax.config.update("jax_enable_x64", True)

from space_groups import SpaceGroup
from space_groups.invariance import construct_diagonal

num_rounds = 5

group_nums = range(1, 231)
num_args = range(2, 5)
supercell_sizes = [(2,2,2), (3,3,3)] # FIXME
test_params = list(product(group_nums, num_args, supercell_sizes))

class TestSpaceDiagonalInvariance(ut.TestCase):

  def setUp(self):
    self.rng = jrnd.PRNGKey(1)

  def rng_split(self):
    self.rng, rng = jrnd.split(self.rng)
    return rng

  def create_invariant_func(
      self, group, rng, supercell, num_args, num_feats=100, lengthscale=0.1,
    ):
    omega_rng = self.rng_split()
    phi_rng = self.rng_split()
    weights_rng = self.rng_split()

    omegas = jrnd.normal(omega_rng, shape=(6*num_args, num_feats)) / lengthscale
    phis = jrnd.uniform(phi_rng, shape=(num_feats,)) * 2 * jnp.pi
    weights = jrnd.normal(weights_rng, shape=(num_feats,)) / jnp.sqrt(num_feats)
    func = lambda x: jnp.cos(x.reshape(-1,6*num_args) @ omegas + phis) @ weights

    return construct_diagonal(func, group, supercell)

  @params(*test_params)
  def test_01_diagonal_translation(self, group_num, num_args, supercell):
    '''Test for invariance to diagonal translations.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split(), supercell, num_args)

    for ii in range(num_rounds):

      # Create a lattice of points consistent with diagonal translations.
      point = jrnd.normal(self.rng_split(), shape=(1, num_args, 3,))
      mesh = jnp.stack(
          jnp.meshgrid(*[jnp.arange(-2, 3) for _ in range(3)]),
          axis=-1,
        ).reshape(-1,3)
      
      pts = mesh[:,jnp.newaxis,:] + point
    
      # Validate that the invariant function is invariant to translations.
      vals = F(pts)
      nptest.assert_allclose(vals, vals[0])

  @params(*test_params)
  def test_02_supercell_translation(self, group_num, num_args, supercell):
    '''Test for invariance to supercell translations.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split(), supercell, num_args)

    for ii in range(num_rounds):
      for jj in range(num_args):

        # Create a lattice of points consistent with the supercell translations.
        point = jrnd.normal(self.rng_split(), shape=(1, num_args, 3,))
        mesh = jnp.stack(
            jnp.meshgrid(*[jnp.arange(-2, 3) for _ in range(3)]),
            axis=-1,
          ).reshape(-1,3)*jnp.array([supercell])
      
      pts = jnp.tile(point, (mesh.shape[0],1,1))

      pts = pts.at[:,jj,:].add(mesh[:,:])
    
      # Validate that the invariant function is invariant to translations.
      vals = F(pts)
      nptest.assert_allclose(vals, vals[0])

  @params(*test_params)
  def test_03_diagonal_action(self, group_num, num_args, supercell):
    '''Test for invariance to diagonal group actions.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split(), supercell, num_args)
    operations = jnp.array([op.to_doubles() for op in G.operations])

    for ii in range(num_rounds):

      # Create a random set of arguments.
      pts = jrnd.normal(self.rng_split(), shape=(100, num_args, 3))

      # Apply the function to the initial values.
      vals0 = F(pts)

      # Put points into homogeneous coordinates.
      pts = jnp.concatenate([pts, jnp.ones((*pts.shape[:2], 1))], axis=-1)

      for op in operations:
        # Apply the operation to the points.
        op_pts = jnp.tensordot(pts, op, axes=(2,1))

        # Trim off the last dimension.
        op_pts = op_pts[...,:3]

        # Apply the invariant function.
        vals = F(op_pts)

        nptest.assert_allclose(vals0, vals)


  @params(*test_params)
  def test_04_separate_action(self, group_num, num_args, supercell):
    '''Test for lack of invariance to separate group actions.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split(), supercell, num_args)

    # Exclude the identity.
    operations = jnp.array([op.to_doubles() for op in G.operations[1:]])

    for ii in range(num_rounds):

      # Create a random set of arguments.
      pts = jrnd.normal(self.rng_split(), shape=(100, num_args, 3))

      # Apply the function to the initial values.
      vals0 = F(pts)

      for jj in range(num_args):

        # Modify the jjth argument.
        jj_args = pts[:,jj,:]

        # Put just these into homogenous coords.
        jj_args = jnp.concatenate([jj_args, jnp.ones((jj_args.shape[0], 1))], axis=-1)

        # Apply the group operation
        for op in operations:
          op_jj_args = jj_args @ op.T
          
          op_pts = pts.at[:,jj,:].set(op_jj_args[...,:3])
    
          vals = F(op_pts)

          self.assertFalse(jnp.any(jnp.isclose(vals0, vals)))

  @params(*test_params)
  def test_05_separate_translation(self, group_num, num_args, supercell):
    '''Test for lack of invariance to separate unit cell translations.'''

    G = SpaceGroup(group_num)
    F = self.create_invariant_func(G, self.rng_split(), supercell, num_args)

    for ii in range(num_rounds):

      # Create a random set of arguments.
      pts = jrnd.normal(self.rng_split(), shape=(100, num_args, 3))

      # Apply the function to the initial values.
      vals0 = F(pts)

      for jj in range(num_args):

        # Loop over dimensions.
        for dd in range(3):

          # Loop over sub-supercell translations.
          for tt in range(1,supercell[dd]):

            # Modify the jjth argument.
            jj_args = pts[:,jj,:]

            op_jj_args = jj_args.at[:,dd].add(tt)

            op_pts = pts.at[:,jj,:].set(op_jj_args)
    
            vals = F(op_pts)

            # rtol because we seem to get unlucky sometimes.
            self.assertFalse(jnp.any(jnp.isclose(vals0, vals, rtol=1e-9)))
