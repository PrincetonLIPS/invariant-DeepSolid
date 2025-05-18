import argparse
import logging
import sys
import re
import pathlib
import jsonpickle
import numpy as np

from cctbx.sgtbx import space_group, space_group_info
from cctbx.sgtbx.direct_space_asu import (
  reference_table as sg_asu_table,
  plane_group_reference_table as pg_asu_table,
)
from cctbx.uctbx import unit_cell

import space_groups.rational_numpy as rnp

from space_groups import PlaneGroup, SpaceGroup, PlaneRegion, SpaceRegion

log = logging.getLogger()

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
    '--data-dir',
    type=str,
    default='space_groups/data',
    help='Directory to store the JSON files.',
  )

  return parser.parse_args()

def generate_space_group_json(idx, data_dir):
  sgi = space_group_info(idx)
  sgt = sgi.type()
  sg  = sgi.group()
  uc  = unit_cell(SpaceGroup.basic_unit_cells[sg.crystal_system()])
  asu = sg_asu_table.get_asu(sgt.number())

  log.info('Generating JSON file for %s.' % (sgi.symbol_and_number()))

  # Create an uninitialized SpaceGroup object.
  SG = SpaceGroup.__new__(SpaceGroup)
  SG.number = idx

  # Basic information.
  # Make sure there aren't annoying characters in the name.
  SG.name = re.sub(
    r'\(.*\)',
    '',
    sgi.symbol_and_number(),
  ).strip().replace(' ', ' ')
  log.debug("Reformatting name to '%s'." % (SG.name))
  SG.is_symmorphic = sgt.is_symmorphic()
  SG.is_enantiomorphic = sgt.is_enantiomorphic()
  SG.crystal_system = sg.crystal_system()

  # Symmetry operations.
  SG.operations = [
    rnp.RationalArray.from_scitbx_matrix(op.as_4x4_rational())
    for op in sg.all_ops(mod=1)
  ]

  # Set up the ASU.
  SG.asu = SpaceRegion.__new__(SpaceRegion)

  SG.asu.vertices = rnp.unique(
    rnp.RationalArray.from_list(asu.shape_vertices())
  )

  normals   = []
  distances = []
  for cut in asu.cuts:
    normals.append(rnp.column_stack(
      [rnp.RationalArray.from_boost_rational(x) for x in cut.n]
    ))
    distances.append(rnp.RationalArray.from_boost_rational(cut.c))
  SG.asu.normals = rnp.row_stack(normals)
  SG.asu.distances = rnp.row_stack(distances)

  # Asymmetric unit bounding box.
  SG.asu.bmin = rnp.RationalArray.from_list(asu.box_min())
  SG.asu.bmax = rnp.RationalArray.from_list(asu.box_max())

  # Wyckoff transformations.
  # http://cci.lbl.gov/cctbx_sources/cctbx/sgtbx/reference_settings/wyckoff.cpp
  SG.wyckoff_positions = []
  wyckoff_table = sgi.wyckoff_table()
  num_wyckoff_pos = wyckoff_table.size()
  for ii in range(num_wyckoff_pos):
    pos = wyckoff_table.position(ii)
    op = rnp.RationalArray.from_scitbx_matrix(
      pos.special_op().as_4x4_rational()
    )
    SG.wyckoff_positions.append({
      'op': op,
      'letter': pos.letter(),
      'multiplicity': pos.multiplicity(),
    })

  # Basic orthogonization matrix.
  SG.basic_basis = np.array(uc.orthogonalization_matrix()).reshape(3,3)

  # Dump to JSON.
  filename = pathlib.Path(data_dir) / SG._filename
  log.debug('Storing to file %s.' % (filename))
  with open(filename, 'w') as fh:
    json_string = jsonpickle.encode(SG)
    fh.write(json_string)

def generate_plane_group_json(idx, data_dir):
  # Find the relevant space group.
  pg_name, sg_name = PlaneGroup.space_group_map[idx]

  sg = space_group(sg_name)
  sgi = space_group_info(idx)
  sgt = sgi.type()
  uc  = unit_cell(SpaceGroup.basic_unit_cells[sg.crystal_system()])
  asu = pg_asu_table.get_asu(idx)

  PG = PlaneGroup.__new__(PlaneGroup)

  # Fix it up to be a plane group.
  PG.name = pg_name
  PG.number = idx

  log.info('Plane group %s (%d) corresponds to space group %s.' \
           % (pg_name, idx, sg_name))

  PG.is_symmorphic = sgt.is_symmorphic()
  PG.is_enantiomorphic = sgt.is_enantiomorphic()
  PG.crystal_system = sg.crystal_system()

  # Symmetry operations.
  PG.operations = [
      rnp.delete(
        rnp.delete(
          rnp.RationalArray.from_scitbx_matrix(op.as_4x4_rational()),
          2, axis=0),
        2, axis=1)
      for op in sg.all_ops(mod=1)
    ]

  # Fix the ASU.
  PG.asu = PlaneRegion.__new__(PlaneRegion)
  PG.asu.vertices = rnp.unique(
    rnp.RationalArray.from_list(asu.shape_vertices())
  )

  normals   = []
  distances = []
  for cut in asu.cuts:
    normals.append(rnp.column_stack(
      [rnp.RationalArray.from_boost_rational(x) for x in cut.n]
    ))
    distances.append(rnp.RationalArray.from_boost_rational(cut.c))
  PG.asu.normals = rnp.row_stack(normals)
  PG.asu.distances = rnp.row_stack(distances)

  # Asymmetric unit bounding box.
  PG.asu.bmin = rnp.RationalArray.from_list(asu.box_min())
  PG.asu.bmax = rnp.RationalArray.from_list(asu.box_max())

  # Remove extra dimension.
  PG.asu.vertices = rnp.unique(PG.asu.vertices[:,:2])
  PG.asu.normals = PG.asu.normals[:,:2]
  PG.asu.bmin = PG.asu.bmin[:2]
  PG.asu.bmax = PG.asu.bmax[:2]

  # Basic orthogonization matrix.
  PG.basic_basis = np.array(uc.orthogonalization_matrix()).reshape(3,3)[:2,:2]

  # Dump to JSON.
  filename = pathlib.Path(data_dir) / PG._filename
  log.debug('Storing to file %s.' % (filename))
  with open(filename, 'w') as fh:
    json_string = jsonpickle.encode(PG)
    fh.write(json_string)

def main():
  args = parse_command_line()

  logging.basicConfig(
    level=args.loglevel,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
  )

  for idx in range(1,231):
    generate_space_group_json(idx, args.data_dir)

  for idx in range(1,18):
    generate_plane_group_json(idx, args.data_dir)

  return 0


if __name__ == '__main__':
  sys.exit(main())
