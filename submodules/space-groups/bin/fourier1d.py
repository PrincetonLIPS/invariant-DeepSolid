import jax
import jax.numpy as jnp
import jax.random as jrnd
import scipy.linalg as sla
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.integrate as spi

jax.config.update("jax_enable_x64", True)

from scipy.special import roots_legendre

from space_groups.invariance import torus

orbifold_dims = 2
num_args = 2
num_basis = 100
num_feats = 10
supercell = 3
lengthscale = 0.5
quad_deg = 100
mirror = True
seed = 1

# Supercell is [0, 1]^2

rng = jrnd.PRNGKey(seed)
rng, omega_rng, phi_rng, weights_rng = jrnd.split(rng, 4)

omegas = jrnd.normal(omega_rng, shape=(num_args, orbifold_dims, num_basis, num_feats)) / lengthscale
phis = jrnd.uniform(phi_rng, shape=(num_args, num_basis, num_feats)) * 2 * jnp.pi
weights = jrnd.normal(weights_rng, shape=(num_args, num_basis, num_feats,)) / jnp.sqrt(num_feats)

x = npr.randn(num_args,1)

@jax.jit
def raw_supercell_basis(x):
  # assume data are num_args x dims and we're going to vmap.

  # push through the orbifold
  # now num_args x (dims * orbifold_dims)
  # torus expects values in [0, 1]
  orb_x = torus(x)

  cos = jnp.cos(jnp.sum(orb_x[:,:,jnp.newaxis, jnp.newaxis] * omegas, axis=1) + phis)
  funcs = jnp.sum(cos * weights, axis=-1)

  return funcs

# Translations are 0, 1, ..., supercell-1.
finite_group_t = jnp.arange(supercell)/supercell
finite_group_a = jnp.ones_like(finite_group_t)
if mirror:
  finite_group_t = jnp.concatenate([finite_group_t, finite_group_t])
  finite_group_a = jnp.concatenate([finite_group_a, -finite_group_a])

@jax.jit
def group_basis(x):
  # this gets used separately when we do it the "fast" way.
  # assume data are num_args x dims and we're going to vmap.

  # expand the data to include the shifts and reflections from the finite group
  aug_x = x[jnp.newaxis,:,:] * finite_group_a[:,jnp.newaxis,jnp.newaxis] \
      + finite_group_t[:,jnp.newaxis,jnp.newaxis]
  
  # vmap over the group
  return jax.vmap(raw_supercell_basis, in_axes=(0,))(aug_x)

# Construct the basis functions and make them invariant.
@jax.jit
def invariant_basis(x):
  per_arg_funcs = group_basis(x)

  per_action_funcs = jnp.prod(per_arg_funcs, axis=1)

  return jnp.sum(per_action_funcs, axis=0)
vmap_invariant_basis = jax.jit(jax.vmap(invariant_basis, in_axes=(0,)))

# Integrate all pairs of basis functions for a given n.
quad_roots, quad_weights = roots_legendre(quad_deg)
quad_roots = (quad_roots + 1) / 2
quad_weights = quad_weights / 2
# quad_roots_Nd = jnp.stack(jnp.meshgrid(*([quad_roots]*num_args)), axis=-1)
# quad_weights_Nd = jnp.outer(quad_weights, quad_weights)
# for ii in range(2, num_args):
#   quad_weights_Nd = jnp.tensordot(quad_weights_Nd, quad_weights, axes=((),()))

# # Evaluate all the basis functions at the quadrature points.
# quad_basis = vmap_invariant_basis(quad_roots_Nd.reshape(-1,num_args,1)).reshape(*([quad_deg]*num_args), num_basis)

# # Take the outer product of the basis functions.
# quad_basis_outer = quad_basis[...,jnp.newaxis,:] * quad_basis[...,:,jnp.newaxis]

# # Weight and sum.
# B = jnp.sum(quad_weights_Nd[...,jnp.newaxis,jnp.newaxis] * quad_basis_outer, axis=range(num_args))
# print('Computed B')

thing = jax.vmap(group_basis, in_axes=(0,))(jnp.tile(quad_roots, (num_args,1)).T[:,:,jnp.newaxis])
thing_outer = thing[:,:,jnp.newaxis,:,:,jnp.newaxis] * thing[:,jnp.newaxis,:,:,jnp.newaxis,:]
thing_int = jnp.sum(thing_outer * quad_weights[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis,jnp.newaxis], axis=0)
thing_prod = jnp.prod(thing_int, axis=2)
thing_B = jnp.sum(thing_prod, axis=(0,1))
# Correct!

################################################################################

# Compute the energy form.
# invariant_basis_grad_func = jax.jit(lambda x: jnp.squeeze(jax.jacfwd(invariant_basis)(x)))
# grads = jax.vmap(invariant_basis_grad_func)(quad_roots_Nd.reshape(-1,num_args,1))
# inners1 = jnp.sum(grads[:,:,jnp.newaxis,:] * grads[:,jnp.newaxis,:,:], axis=-1)
# inners = jnp.reshape(inners1, (*([quad_deg]*num_args), num_basis, num_basis))
# A = jnp.sum(quad_weights_Nd[...,jnp.newaxis,jnp.newaxis] * inners, axis=range(num_args))
# print('Computed A')

# jacfwd is output x input

# Hacky with the diagonal.  I think this is right.
diag_jac = lambda x: jnp.squeeze(jax.jacfwd(group_basis)(x))[:,jnp.arange(num_args),:,jnp.arange(num_args)]

# This is just evaluating the gradients at the quadrature points.
# Let's put this in the order that I have in the math.
d_group_basis = jax.jit(jax.vmap(diag_jac))(jnp.tile(quad_roots, (num_args,1)).T[:,:,jnp.newaxis])
d_group_basis = jnp.moveaxis(d_group_basis, [0,1,2,3], [2,1,0,3])

# Take all of the inner products of the gradients.
d_group_inners = d_group_basis[:,jnp.newaxis,:,:,:,jnp.newaxis] * d_group_basis[jnp.newaxis,:,:,:,jnp.newaxis,:]

# Perform the integration.
d_group_inner_int = jnp.sum(d_group_inners * quad_weights[:,jnp.newaxis,jnp.newaxis], axis=3)

# Reverse the order of the argument axis and then multiply.
A_sum = jnp.sum(d_group_inner_int * thing_int[:,:,::-1,:,:], axis=2)

thing_A = jnp.sum(A_sum, axis=(0,1))

################################################################################
# Solve the generalized eigenvalue problem.
eigvals, eigvecs = sla.eigh(thing_A, thing_B)
print('Solved eigenvalue problem.')

rez = 250
#Plot the results.
pxx = jnp.linspace(-1, 1, rez)
pxy = jnp.linspace(-1, 1, rez)
px_grid = jnp.stack(jnp.meshgrid(pxx, pxy), axis=-1)

basis = vmap_invariant_basis(px_grid.reshape(-1,2,1)).reshape(rez,rez,num_basis)

plt.figure(1)
for ii in range(16):
  efunc = basis @ eigvecs[:,ii]
  plt.subplot(4,4,ii+1)
  plt.contourf(px_grid[:,:,0], px_grid[:,:,1], jnp.reshape(efunc, (rez,rez)))
  plt.gca().set_aspect('equal')
  plt.gca().set_title('%0.4f' % (eigvals[ii]/(4*jnp.pi**2)))

#plt.figure(2)
#for ii in range(16):
#  plt.subplot(4,4,ii+1)
#  plt.contourf(px_grid[:,:,0], px_grid[:,:,1], basis[:,:,ii])
#  plt.gca().set_aspect('equal')

# plt.figure(3)
# pts = jnp.arange(-5, 6)
# ptx, pty = jnp.meshgrid(pts, pts)
# pt_grid = jnp.stack([ptx, pty], axis=-1).reshape(-1,2)
# for ii, pt in enumerate(pt_grid):
#   if jnp.sum(pt) % 3 == 0:
#     plt.plot(pt[0], pt[1], 'o', color='black')
#     plt.text(pt[0], pt[1], '%d' % (jnp.sum(pt**2)))
#   else:
#     plt.plot(pt[0], pt[1], '.', color='gray')
# plt.grid(True)
# plt.gca().set_aspect('equal')

plt.show()
