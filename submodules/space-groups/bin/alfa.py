import argparse
import logging
import sys

from space_groups import PlaneGroup, SpaceGroup

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

  return parser.parse_args()

def main():
  args = parse_command_line()

  logging.basicConfig(
    level=args.loglevel,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
  )

  for idx in range(1,231):
    sg = SpaceGroup(idx)
    log.info('Loaded space group %s (%d).' % (sg.name, idx))
    print(sg.operations)

  for idx in range(1,17):
    pg = PlaneGroup(idx)
    log.info('Loaded plane group %s (%d).' % (pg.name, idx))
    print(type(pg.operations[0]))
    #print(pg.asu.vertices)
    #print(pg.asu.normals)

  return 0

if __name__ == '__main__':
  sys.exit(main())
