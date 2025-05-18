# space-groups
Utility library for manipulating wallpaper and space groups.

## Plane and space group information
Using the basic functionality looks like this:
```python
from space_groups import PlaneGroup

# Get p4.
G = PlaneGroup(10)

# Iterate over the operations.
for op in G.operations:

  # Cast from RationalNumpy to regular Numpy.
  double_op = op.to_doubles()

  # Do a thing with the operation ...

# Do things with the fundamental region.
vertices = G.asu.vertices
```

## Single-argument invariance
To construct a function that is invariant in a single argument, define a
function that takes a batch of vectors with twice as many dimensions as the
space you're operating in, i.e., ndarrays of shape (batch_size x 2*num_dims)
where num_dims=2 for plane groups and num_dims=3 for space groups.  Then
pass this function to the `construct_single` function along with the group
you care about:
```python
from space_groups.invariance import construct_single

# Get p4.
G = PlaneGroup(10)

# Here's a function that can take four dimensions.
func = lambda x: jnp.sum(x**2, axis=-1)

# Here's a p4-invariant version of that function.
invariant_func = construct_single(func, G)
```

## Diagonal invariance with multiple arguments.
To construct a function that has diagonal invariance across multiple arguments,
define a function that takes as input ndarrays of size
(batch_size x num_args x 2*num_dims) where num_dims=2 for plane groups and
num_dims=3 for space groups as above.  Then call `construct_diagonal` on this
function along with the group and the relevant supercell size specified as a
tuple.  It is assumed that the primitive cells are 1x1x1 and so the supercells
must be tuples of integers like (3,4,5), although they will need to be equal
when the group has rotations.
```python
from space_groups.invariance import construct_diagonal

# Get p4.
G = PlaneGroup(10)

# Here's a function that can take three arguments each with four dimensions.
func = lambda x: jnp.sum(x ** jnp.arange(3)[jnp.newaxis,:,jnp.newaxis], axis=-1)

# Here's a diagonally p4-invariant version of that function that is also
# invariant to separate supercell translations of size 3.
invariant_func = construct_diagonal(func, G, (3,3,3))
```

## Running tests
Run tests with `nose2 -vv --fail-fast` or equivalent.

Can also parallel testing with something like
`nose2 -vv --fail-fast --plugin=nose2.plugins.mp -N 2`