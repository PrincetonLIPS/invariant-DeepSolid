import jax
import jax.numpy as jnp

from .orbifold import torus

def construct_diagonal(fun, group, supercell=None):
  '''Constructs a diagonally-invariant function for a crystallographic group.

  Diagonal invariance is the property that the function is invariant to
  applying the same operation to all arguments.  For example, if we have a group
  G then a function f is diagonally invariant if:

    f(g(x_1), g(x_2), ...) = f(x_1, x_2, ...) for all g in G.

  Optionally, a supercell may be specified and the function will be made
  separately invariant to supercell shifts.  That is, the function will be
  periodic with respect to the supercell.

  Arguments:
  ----------
    fun : Callable[[jnp.ndarray, ...], jnp.ndarray]
      This is the function to be made invariant.  It should take as its first
      argument a JAX ndarray with shape (num_args, num_dims*2), where num_dims
      is, e.g., 2 for wallpaper groups and 3 for space groups. It can optionally
      take *args and **kwargs as additional arguments that will be passed
      through to the function.  It should be vmappable.

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

    supercell : Optional[Tuple[int, ...]]
      This is the supercell to which the function should be made separately
      invariant.  If None, then the function will not be periodic and will
      only by diagonally invariant to the specified group.

  Returns:
  --------
    invariant_f : Callable[[jnp.ndarray, ...], jnp.ndarray]
  '''

  num_dims = group.dims
  operations = jnp.array([op.to_doubles() for op in group.operations])
  num_ops = operations.shape[0]

  if supercell is not None:
    assert len(supercell) == num_dims

    # TODO: Verify that the group is compatible with the supercell.

    # Get all the translations in the supercell.
    grids = [jnp.arange(k) for k in supercell]
    translations = jnp.stack(jnp.meshgrid(*grids), axis=-1).reshape(-1,num_dims)
    periods = jnp.array(supercell)

  else:

    # FIXME: This is actually wrong.  If there is no supercell then we don't
    # push it through the orbifold.
    translations = jnp.zeros((1,2))
    periods = jnp.ones((2,))

  def invariant_f(x, *args, **kwargs):
    x = jnp.atleast_3d(x)

    # x is assumed to be of shape (batch_sz, num_args, num_dims)
    batch_sz = x.shape[0]
    num_args = x.shape[1]

    # Add a constant dimension to make it batch_sz x num_args x (num_dims+1)
    proj_x = jnp.concatenate([x, jnp.ones((*x.shape[:2], 1))], axis=-1)
    
    # vmap over both the number of operations and the batch size, applying the
    # operations. Becomes batch_sz x num_ops x num_args x (num_dims+1).
    op_x = jax.vmap(
      jax.vmap(
          lambda op, x: x @ op.T,
          in_axes=(0, None),
        ),
      in_axes=(None, 0)
      )(operations, proj_x)

    # Trim off the last dimension and reshape to batch_sz x num_ops x num_args x num_dims.
    op_x = op_x[...,:num_dims]

    # Put the op dimension first.
    op_x = op_x.transpose(1,0,2,3)

    # Apply all translations and roll in with the ops.
    top_x = (
      op_x[jnp.newaxis,:,:,:,:]
        + translations[:,jnp.newaxis,jnp.newaxis,jnp.newaxis,:]
      ).reshape(-1, *op_x.shape[1:])

    # Push through the orbifold map, vmapping over all but the last dimension.
    orbifold_x = jax.vmap(
      jax.vmap(
        jax.vmap(
          torus, 
          in_axes=(0,),
        ),
        in_axes=(0,),
      ),
      in_axes=(0,),
      )(top_x/periods[jnp.newaxis,jnp.newaxis,:])

    # Apply function, vmapping over batch and operations.
    # becomes batch_sz x num_ops
    fun_vals = jax.vmap(
      jax.vmap(
        lambda x: fun(x, *args, **kwargs),
        in_axes=(0,),
      ),
      in_axes=(0,),
      )(orbifold_x)

    # Average over the operation dimension.
    return jnp.mean(fun_vals, axis=0).ravel()
  
  return invariant_f

