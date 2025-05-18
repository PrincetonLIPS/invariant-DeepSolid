import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.numpy.linalg as jnpla
import matplotlib.pyplot as plt

from space_groups import PlaneGroup
from space_groups.invariance import construct_diagonal_2d

group = 16
num_args = 3
seed = 1
lengthscale = 0.5
num_feats = 100

pxx = jnp.linspace(-2, 2, 500)
pxy = jnp.linspace(-2, 2, 500)
px_grid = jnp.stack(jnp.meshgrid(pxx, pxy), axis=-1)

rng = jrnd.PRNGKey(seed)
rng, omega_rng, phi_rng, weights_rng = jrnd.split(rng, 4)

omegas = jrnd.normal(omega_rng, shape=(4*num_args, num_feats)) / lengthscale
phis = jrnd.uniform(phi_rng, shape=(num_feats,)) * 2 * jnp.pi
weights = jrnd.normal(weights_rng, shape=(num_feats,)) / jnp.sqrt(num_feats)
func = lambda x: jnp.cos(x.reshape(-1,4*num_args) @ omegas + phis) @ weights

G = PlaneGroup(group)

invariant_func = construct_diagonal_2d(func, G)

#plt.contourf(px_grid[:,:,0], px_grid[:,:,1], invariant_func(px_grid.reshape(-1,2) @ jnpla.inv(G.basic_basis).T).reshape(500,500))
#plt.gca().set_aspect('equal')
#plt.show()

import numpy.random as npr
X = npr.randn(5, num_args, 2)

F0 = invariant_func(X)

# Validate that the invariant function is not invariant separately.
for op in G.operations[1:]:
  op = op.to_doubles()

  # Apply ops to individual arguments.
  for ii in range(num_args):
    Y = jnp.concatenate([X, jnp.ones((*X.shape[:2], 1))], axis=-1)
    Y = Y.at[:,ii,:].set(Y[:,ii,:] @ op.T)[:,:,:2]
    F1 = invariant_func(Y)
    assert not jnp.allclose(F0, F1)

# Validate that the invariant function is diagonally invariant
for op in G.operations[1:]:
  op = op.to_doubles()

  # Apply ops to all arguments.
  Y = jnp.concatenate([X, jnp.ones((*X.shape[:2], 1))], axis=-1)
  Y = jnp.tensordot(Y, op.T, axes=(2,0))[:,:,:2]

  F1 = invariant_func(Y)
  assert jnp.allclose(F0, F1)
