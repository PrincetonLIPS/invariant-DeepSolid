import numpy as np
import numpy.random as npr

from functools import cached_property
from scipy.spatial import ConvexHull

from .. import rational_numpy as rnp

class FundamentalRegion:
  ''' Class representing the fundamental region of space and plane groups.

  This fundamental region is represented as a convex shape.

  Attributes:
  -----------

    vertices : RationalArray
      A rational array containing all the vertices of the convex polytope.

    normals : RationalArray
      A rational array containing all of the normals for the faces of the convex
      polytope.  These are computed from the convex hull.

    hull : scipy.spatial.ConvexHull
      This is the convex hull representation for the polytope.

    volume : float
      In 3d this is the volume and in 2d this is the area of the fundamental
      region.

    bmin : RationalArray
      The smaller corner of a bounding box for the fundamental region.

    bmax : RationalArray
      The upper corner of a bounding vox for the fundamental region.
  '''

  @cached_property
  def hull(self):
    return ConvexHull(self.vertices.to_doubles())

  @cached_property
  def volume(self):
    return self.hull.volume

  def is_inside(self, points):
    ''' Check if a point is inside the fundamental region.'''

    # Extend points with a column of ones for the offset in the plane equation
    extended_points = np.hstack([points, np.ones((points.shape[0], 1))])

    # Check if all facets' equations are satisfied for each point
    return np.all(np.dot(self.hull.equations, extended_points.T) <= 0, axis=0)
  
  def random_points(self, num_points, rng=None):
    ''' Generate random points inside the fundamental region.

    Parameters:
    -----------

      num_points : int
        The number of points to generate.

    Returns:
    --------

      points : numpy.ndarray
        A numpy array containing the points.
    '''

    # Generate more points in the bounding box than we need.
    overshoot = 2

    bmin = self.bmin.to_doubles()
    bmax = self.bmax.to_doubles()

    # Generate random points in the bounding box.
    if rng is None:
      points = npr.rand(overshoot * num_points, self.dim) * (bmax - bmin) + bmin
    else:
      points = rng.rand(overshoot * num_points, self.dim) * (bmax - bmin) + bmin

    # Keep only the points that are inside the fundamental region.
    points = points[self.is_inside(points)]

    # If we don't have enough points, try again.
    if len(points) < num_points:
      return np.vstack([points, self.random_points(num_points - len(points), rng=rng)])
    else:
      return points[:num_points,:]
