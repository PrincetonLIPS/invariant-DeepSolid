import jax
import jax.numpy as jnp

from .orbifold import torus

def construct_single(fun, group):
  '''Construct a singly-invariant function for a crystallographic group.

  Arguments:
  ----------
    fun : Callable[[jnp.ndarray, ...], jnp.ndarray]
      This is the function to be made invariant.  It should take as its first
      argument a batch of points of shape (batch_size, num_dims*2) where
      num_dims is, e.g., 2 for wallpaper groups and 3 for space groups.  It
      can optionally take *args and **kwargs as additional arguments that will
      be passed through to the function.

      It is assumed that you are handling the basis on your own and the basis
      this function sees is the standard basis.  That is, the translation
      subgroup is always unit translations.  This means, for example, that if
      you have trigonal or hexagonal symmetry, you will need to transform
      the inputs to the invariant function via something like:

        invariant_f( x @ jnpla.inv(group.basic_basis).T )

    group : SymmetryGroup
      This is the group to which the function should be made invariant.  For
      two dimensional functions this should be a PlaneGroup, and for three
      dimensional functions this should be a SpaceGroup.

  Returns:
  --------
    invariant_f : Callable[[jnp.ndarray, ...], jnp.ndarray]
  '''

  num_dims = group.dims
  operations = jnp.array([op.to_doubles() for op in group.operations])
  num_ops = operations.shape[0]

  def invariant_f(x, *args, **kwargs):

    # Append ones to use homogenous coordinates.
    proj_x = jnp.hstack([x, jnp.ones((x.shape[0], 1))])

    # Apply all the operations.
    op_x = jax.vmap(
      lambda op, x: x @ op.T,
      in_axes=(0, None),
      )(operations, proj_x)

    # Reshape to (batch_sz x num_ops x num_args) x num_dims and project out of
    # homogenous coordinates by trimming off the last dimension.
    op_x = jnp.swapaxes(op_x, 0, 1).reshape((-1, num_dims+1))[:,:num_dims]

    # Apply the orbifold map.
    orbifold_x = jax.vmap(torus, in_axes=(0,))(op_x)

    # Evaluate the function on the orbifold embedding.
    # Should end up with something batch_sz x num_ops.
    fun_vals = jnp.reshape(fun(orbifold_x, *args, **kwargs), (-1, num_ops))

    # Average over the operation dimension.
    return jnp.mean(fun_vals, axis=-1)
  
  return invariant_f

