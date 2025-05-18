from setuptools import setup

setup(
  name='space_groups',
  version='0.1.0',
  description='Simple python package for wallpaper and space groups',
  url='https://github.com/PrincetonLIPS/space-groups',
  author='Ryan P. Adams',
  author_email='rpa@princeton.edu',
  license='BSD 2-clause',
  packages=['space_groups'],
  package_dir={'space_groups': 'space_groups'},
  package_data={'space_groups': ['data/*.json']},
  install_requires=[
    'numpy',
    'jsonpickle',
  ],
  classifiers=[], # FIXME
)
