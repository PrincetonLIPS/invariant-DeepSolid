import jax
import jax.numpy as jnp
import jax.random as jrnd
import jax.numpy.linalg as jnpla
import matplotlib.pyplot as plt

from space_groups import PlaneGroup
from space_groups.invariance import construct_single

group = 16
seed = 3
lengthscale = 0.5
num_feats = 100

pxx = jnp.linspace(-2, 2, 500)
pxy = jnp.linspace(-2, 2, 500)
px_grid = jnp.stack(jnp.meshgrid(pxx, pxy), axis=-1)

rng = jrnd.PRNGKey(seed)
rng, omega_rng, phi_rng, weights_rng = jrnd.split(rng, 4)

omegas = jrnd.normal(omega_rng, shape=(4, num_feats)) / lengthscale
phis = jrnd.uniform(phi_rng, shape=(num_feats,)) * 2 * jnp.pi
weights = jrnd.normal(weights_rng, shape=(num_feats,)) / jnp.sqrt(num_feats)
func = lambda x: jnp.cos(x @ omegas + phis) @ weights

G = PlaneGroup(group)

invariant_func = construct_single(func, G)

vals = invariant_func(px_grid.reshape(-1,2) @ jnpla.inv(G.basic_basis).T)

plt.contourf(px_grid[:,:,0], px_grid[:,:,1], vals.reshape(500,500))
plt.gca().set_aspect('equal')
plt.show()
