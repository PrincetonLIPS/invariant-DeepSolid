import numpy as np

# TODO: probably need these to work with rational numpy also.

def up(points):
  return np.hstack([points, np.ones(points.shape[:-1]+(1,))])

def dn(points):
  return points[...,:-1]
