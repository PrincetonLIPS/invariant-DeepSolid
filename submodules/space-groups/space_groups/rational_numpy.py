import numpy as np

from functools import partial

def recursive_map(func, items, tolist=True):
  newfunc = lambda x: recursive_map(func, x, tolist=tolist) \
    if type(x) in (list, tuple) else func(x)
  if tolist:
    return list(map(newfunc, items))
  else:
    return map(newfunc, items)

def _simplify(numers, denoms):
  gcds = np.gcd(numers, denoms)

  # Make things zero as appropriate.
  new_numers = np.where(numers == 0, 0, numers // gcds)
  new_denoms = np.where(numers == 0, 1, denoms // gcds)

  # If denominator is negative, multiply by -1.
  new_numers = np.where(new_denoms < 0, -new_numers, new_numers)
  new_denoms = np.where(new_denoms < 0, -new_denoms, new_denoms)

  return new_numers, new_denoms

def scalar_add(r1, c):
  return RationalArray(c*r1.denoms + r1.numers, r1.denoms)

def add(r1, r2):
  new_numers1 = r1.denoms * r2.numers
  new_numers2 = r1.numers * r2.denoms
  sum_numers  = new_numers1 + new_numers2
  sum_denoms  = r1.denoms * r2.denoms
  return RationalArray(sum_numers, sum_denoms)

def scalar_multiply(r1, c):
  return RationalArray(c * r1.numers, r1.denoms)

def multiply(r1, r2):
  new_numers = r1.numers * r2.numers
  new_denoms = r1.denoms * r2.denoms
  return RationalArray(new_numers, new_denoms)

def scalar_divide(r1, c):
  return RationalArray(r1.numers, c*r1.denoms)

def divide(r1, r2):
  new_numers = r1.numers * r2.denoms
  new_denoms = r1.denoms * r2.numers
  return RationalArray(new_numers, new_denoms)

def sum(r1, axis=None, keepdims=False):
  # Get products of all other denominators, along relevant axis.
  prods = np.prod(r1.denoms, axis=axis, keepdims=True) // r1.denoms

  # Multiply by numerators.
  new_numers = r1.numers * prods

  # Sum numerators over relevant axis.
  numer_sums = np.sum(new_numers, axis=axis, keepdims=keepdims)

  # Would be nice to avoid doing the product twice.
  # Just not sure how to handle keepdims.
  return RationalArray(
    numer_sums,
    np.prod(r1.denoms, axis=axis, keepdims=keepdims),
  )

def rationalize(flts, decimals=10):
  scale1 = 10**decimals
  scale2 = 10**(decimals-1)
  numers = np.round(flts * scale1 - flts*scale2).astype(np.int64)
  denoms = np.ones_like(numers) * (scale1 - scale2)
  return RationalArray(numers, denoms)

def arange(*args):
  numers = np.arange(*args, dtype=np.int64)
  denoms = np.ones_like(numers)
  return RationalArray(numers, denoms)

def meshgrid(*args):
  args_numers = [arg.numers for arg in args]
  args_denoms = [arg.denoms for arg in args]
  numer_mesh  = np.meshgrid(*args_numers)
  denom_mesh  = np.meshgrid(*args_denoms)
  return [RationalArray(n,d) for (n,d) in zip(numer_mesh, denom_mesh)]

def stack(arrays, axis=0, out=None):
  numer_stack = np.stack([a.numers for a in arrays], axis=axis, out=out)
  denom_stack = np.stack([a.denoms for a in arrays], axis=axis, out=out)
  return RationalArray(numer_stack, denom_stack)

def concatenate(arrays, **kwargs):
  numer_concat = np.concatenate([a.numers for a in arrays], **kwargs)
  denom_concat = np.concatenate([a.denoms for a in arrays], **kwargs)
  return RationalArray(numer_concat, denom_concat)

def reshape(a, newshape, order='C'):
  return RationalArray(
    np.reshape(a.numers, newshape, order=order),
    np.reshape(a.denoms, newshape, order=order),
  )

def expand_dims(arr, axis):
  return RationalArray(
    np.expand_dims(arr.numers, axis),
    np.expand_dims(arr.denoms, axis),
  )

def delete(arr, obj, axis=None):
  return RationalArray(
    np.delete(arr.numers, obj, axis=axis),
    np.delete(arr.denoms, obj, axis=axis),
  )

# Not a real implementation.  Just does it along the first dimension.
# Could adapt this into a richer thing by rotating that dimension.
def unique(array):
  float_array = array.to_doubles()
  _, indices  = np.unique(float_array, axis=0, return_index=True)
  return array[indices,:]

def is_positive(r1):
  return r1.numers > 0

def is_nonnegative(r1):
  return r1.numers >= 0

def gt(r1, r2):
  return is_positive(r1-r2)

def gte(r1, r2):
  return is_nonnegative(r1-r2)

def lt(r1, r2):
  return is_positive(r2-r1)

def lte(r1, r2):
  return is_nonnegative(r2-r1)

def eq(r1, r2):
  return np.logical_and(r1.numers == r2.numers, r1.denoms == r2.denoms)

def neq(r1, r2):
  return np.logical_not(eq(r1, r2))

def ones(shape):
  return RationalArray(
    np.ones(shape, dtype=np.int64),
    np.ones(shape, dtype=np.int64),
  )

def zeros(shape):
  return RationalArray(
    np.zeros(shape, dtype=np.int64),
    np.ones(shape, dtype=np.int64),
  )

def row_stack(arrays):
  return RationalArray(
    np.row_stack([a.numers for a in arrays]),
    np.row_stack([a.denoms for a in arrays]),
  )

def column_stack(arrays):
  return RationalArray(
    np.column_stack([a.numers for a in arrays]),
    np.column_stack([a.denoms for a in arrays]),
  )

def _dot_1d_1d(r1, r2):
  return sum(multiply(r1, r2), None, False)

def _dot_2d_1d(r1, r2):
  return concatenate([_dot_1d_1d(r,r2) for r in r1])

#_dot_2d_1d = np.vectorize(_dot_1d_1d, signature='(m,n),(n)->(m)')
_dot_1d_2d = np.vectorize(_dot_1d_1d, signature='(n),(n,m)->(m)')
_dot_2d_2d = np.vectorize(_dot_1d_1d, signature='(m,n),(n,k)->(m,k)')

def dot(r1,r2):
  if len(r1.shape) == 2:
    if len(r2.shape) == 2:
      return _dot_2d_2d(r1, r2).T
    elif len(r2.shape) == 1:
      return _dot_2d_1d(r1, r2)
    else:
      return NotImplemented
  elif len(r1.shape) == 1:
    if len(r2.shape) == 2:
      return _dot_1d_2d(r1, r2)
    elif len(r2.shape) == 1:
      return _dot_1d_1d(r1, r2)
    else:
      return NotImplemented
  else:
    return NotImplemented

class RationalArray:
  int64_min = np.iinfo(np.int64).min
  int64_max = np.iinfo(np.int64).max

  def __init__(self, numers, denoms=None):
    denoms = np.ones_like(numers) if denoms is None else denoms

    if type(numers) in (int,) and type(denoms) in (int,):
      numers = np.array([numers])
      denoms = np.array([denoms])
    elif type(numers) in (list,):
      numers = np.array(numers)

    if numers.shape != denoms.shape:
      raise Exception("Numerator and denominator shapes must match.")

    assert not np.any(denoms == 0)

    numers, denoms = _simplify(
      numers.astype(np.int64),
      denoms.astype(np.int64),
    )

    self.numers = numers
    self.denoms = denoms

  def __repr__(self):
    flat_numers = np.array(self.numers.ravel())
    flat_denoms = np.array(self.denoms.ravel())

    strings = np.where(
      flat_numers == 0,
      '0',
      np.where(
        flat_denoms == 1,
        flat_numers.astype(str),
        np.char.add(
          np.char.add(
            flat_numers.astype(str),
            '/',
          ),
          flat_denoms.astype(str),
        ),
      ),
    )

    return np.array2string(
      strings.reshape(self.shape),
      formatter={'str_kind': lambda x: x},
    )

  def __iter__(self):
    idx = 0
    while idx < self.numers.shape[0]:
      yield RationalArray(self.numers[idx,...], self.denoms[idx,...])
      idx += 1

  @classmethod
  def from_scitbx_matrix(cls, mat):
    shape = mat.n

    numers = np.array(
      recursive_map(lambda x: x.numerator(), mat),
      dtype=np.int64,
    ).reshape(shape)
    denoms = np.array(
      recursive_map(lambda x: x.denominator(), mat),
      dtype=np.int64,
    ).reshape(shape)

    return cls(numers, denoms)

  @classmethod
  def from_boost_rational(cls, val):
    if type(val) in (int,):
      numers = np.array(val, dtype=np.int64)
      denoms = np.array(1, dtype=np.int64)
    else:
      numers = np.array(val.numerator(), dtype=np.int64)
      denoms = np.array(val.denominator(), dtype=np.int64)

    return cls(numers, denoms)

  @classmethod
  def from_list(cls, mat):

    numers = np.array(
      recursive_map(lambda x: x.numerator(), mat),
      dtype=np.int64,
    )
    denoms = np.array(
      recursive_map(lambda x: x.denominator(), mat),
      dtype=np.int64,
    )

    return cls(numers, denoms)

  def to_doubles(self):
    return self.numers/self.denoms

  def to_boost(self):
    return NotImplemented

  def __getitem__(self, key):
    return type(self)(self.numers[key], self.denoms[key])

  def add(self, other):
    if type(other) in (int,):
      return scalar_add(self, other)
    elif isinstance(other, type(self)):
      return add(self, other)
    else:
      return NotImplemented
  __add__  = add
  __radd__ = add

  def subtract(self, other):
    return self.add(other * -1)
  __sub__ = subtract

  def multiply(self, other):
    if type(other) in (int,):
      return scalar_multiply(self, other)
    elif isinstance(other, type(self)):
      return multiply(self, other)
    else:
      return NotImplemented
  __mul__  = multiply
  __rmul__ = multiply

  def neg(self):
    return scalar_multiply(self, -1)
  __neg__ = neg

  def divide(self, other):
    if type(other) in (int,):
      return scalar_divide(self, other)
    elif isinstance(other, type(self)):
      return divide(self, other)
    else:
      return NotImplemented
  __truediv__ = divide

  def eq(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return eq(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return eq(self, other)
    else:
      return NotImplemented
  __eq__ = eq

  def neq(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return neq(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return neq(self, other)
    else:
      return NotImplemented
  __ne__ = neq

  def lt(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return lt(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return lt(self, other)
    else:
      return NotImplemented
  __lt__ = lt

  def lte(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return lte(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return lte(self, other)
    else:
      return NotImplemented
  __le__ = lte

  def gt(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return gt(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return gt(self, other)
    else:
      return NotImplemented
  __gt__ = gt

  def gte(self, other):
    if type(other) in (int,):
      ones = np.ones_like(self.numers)
      return gte(self, RationalArray(other * ones, ones))
    elif isinstance(other, type(self)):
      return gte(self, other)
    else:
      return NotImplemented
  __ge__ = gte

  def dot(self, other):
    if type(other) in (int,):
      return self.scalar_multiply(other)
    elif isinstance(other, type(self)):
      return dot(self, other)
    else:
      return NotImplemented
  __matmul__ = dot

  def sum(self, axis=None, keepdims=False):
    return sum(self, axis=axis, keepdims=keepdims)

  def reshape(self, *newshape):
    return reshape(self, newshape)

  @property
  def T(self):
    return type(self)(self.numers.T, self.denoms.T)

  @property
  def shape(self):
    return self.numers.shape
