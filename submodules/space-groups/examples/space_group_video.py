# Make a video of a space group.
# Run this like:
#  $ python space_group_video.py --group 205 --file video.mp4
#

import argparse
import logging
import sys
import numpy as np
import pyvista as pv
import matplotlib as mpl
import tqdm

from scipy.spatial import ConvexHull

from space_groups import PlaneGroup, SpaceGroup

log = logging.getLogger()

logging.getLogger('matplotlib').setLevel('WARNING')
logging.getLogger('absl').setLevel('WARNING')
logging.getLogger('numba').setLevel('WARNING')


def parse_command_line():
  parser = argparse.ArgumentParser(
    description='''
    Generate data files for the space_groups package.
    '''
  )

  parser.add_argument(
    '--loglevel',
    type=str,
    default='WARNING',
    help='DEBUG, INFO, WARNING (default), ERROR, or CRITICAL',
  )

  parser.add_argument(
    '--group',
    type=int,
    default=0,
    help='The space group to render.',
  )

  parser.add_argument(
    '--scale',
    type=float,
    default=0.4,
  )

  parser.add_argument(
    '--file',
    type=str,
    help='The filename for the output movie.',
  )

  return parser.parse_args()


def main():
  args = parse_command_line()

  logging.basicConfig(
    level=args.loglevel,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
  )

  sg = SpaceGroup(args.group)
  basis = np.eye(4)
  basis[:3,:3] = sg.basic_basis

  hull = sg.asu.hull
  faces = np.column_stack((3*np.ones((len(hull.simplices), 1), dtype=np.int32),
                           hull.simplices)).flatten()
  fundamental = pv.PolyData(hull.points, faces)
  fund_vol = hull.volume

  # Asymmetric object, make unit size.
  thing = pv.read("space_groups/data/asymmetric1.stl")
  thing_hull = ConvexHull(thing.points)
  scale = (thing_hull.volume / fund_vol) ** (1/3)
  thing.scale(args.scale/scale, inplace=True)
  thing.translate(-thing.center_of_mass(), inplace=True)
  thing.translate(fundamental.center_of_mass(), inplace=True)

  plot = pv.Plotter(window_size=([2048, 1536]), off_screen=True)
  plot.set_position([5, 5, 5])
  plot.set_background(color='white')

  plot.open_movie(args.file)
  plot.add_text('%s (#%d)' % (sg.name, sg.number),
                position='upper_right', color='black',
                shadow=True, font_size=32)
  #plot.add_text('Ryan P. Adams',
  #              position='lower_left', color='gray',
  #              shadow=False, font_size=12)

  cmap = mpl.cm.get_cmap('tab20')

  objects = []
  centers = []
  for ii, op in enumerate(sg.operations):
    op = op.to_doubles()
    t_fund  = fundamental.transform(basis @ op, inplace=False)
    t_thing = thing.transform(basis @ op, inplace=False)

    color = cmap(ii/(len(sg.operations)-1))

    objects.append([t_fund, t_thing, color])

    plot.add_mesh(t_fund, opacity=0.5, color=color)
    plot.add_mesh(t_thing, color=color)

    centers.append(t_fund.center_of_mass())
  center = np.mean(centers, axis=0)

  for lst in objects:
    direction = lst[0].center_of_mass() - center
    direction = direction / np.sqrt(np.sum(direction**2) + 1e-8)
    lst.append(direction)

  plot.camera.focal_point = center

  plot.show(auto_close=False)
  plot.write_frame()

  az_per_frame = 5
  for ii in tqdm.tqdm(range(360//az_per_frame)):
    plot.camera.azimuth += az_per_frame
    plot.write_frame()

  step_per_frame = 0.02
  for ii in tqdm.tqdm(range(360//az_per_frame)):
    for fund, thing, color, direction in objects:
      t_fund = fund.translate(direction*step_per_frame, inplace=False)
      t_thing = thing.translate(direction*step_per_frame, inplace=False)

      fund.overwrite(t_fund)
      thing.overwrite(t_thing)

      #plot.add_mesh(t_fund, color=color, opacity=0.5)
      #plot.add_mesh(t_thing, color=color)

    plot.camera.azimuth += az_per_frame
    plot.write_frame()

  for ii in tqdm.tqdm(range(360//az_per_frame)):
    for fund, thing, color, direction in objects:
      t_fund = fund.translate(-direction*step_per_frame, inplace=False)
      t_thing = thing.translate(-direction*step_per_frame, inplace=False)

      fund.overwrite(t_fund)
      thing.overwrite(t_thing)

      #plot.add_mesh(t_fund, color=color, opacity=0.5)
      #plot.add_mesh(t_thing, color=color)

    plot.camera.azimuth += az_per_frame
    plot.write_frame()

  for ii in tqdm.tqdm(range(360//az_per_frame)):
    plot.camera.azimuth += az_per_frame
    plot.write_frame()

  plot.close()
  return 0

if __name__ == '__main__':
  sys.exit(main())
