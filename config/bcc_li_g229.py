# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from DeepSolid import base_config
from DeepSolid import supercell
from DeepSolid.utils import poscar_to_cell
import numpy as np
from utils.group_utils import get_group


def get_config(base_cfg, cfg_str='config/poscar/bcc_li.vasp,2,ccpvdz'):
    poscar_path, S, basis = cfg_str.split(',')
    cell = poscar_to_cell.read_poscar(poscar_path)
    L_Bohr = cell.atom[1][1][0]
    S = int(S)
    S = np.diag([S, S, S])
    cell.verbose = 5
    cell.basis = basis
    cell.exp_to_discard = 0.1
    cell.build()
    cfg = base_cfg

    # Set up cell

    simulation_cell = supercell.get_supercell(cell, S)
    if cell.spin != 0:
        simulation_cell.hf_type = 'uhf'
    cfg.system.pyscf_cell = simulation_cell
    cfg.system.pyscf_cell.L_Bohr = L_Bohr

    
    # group used for orbifold layer
    group = get_group(
        plane_or_space='space',
        group_index=229,
        unit_cell=cell.a,
        S=np.diagonal(S),
        L_Bohr=L_Bohr,
        translate=True,
    )

    # group used for performing averaging
    cfg.symmetria.gpave.group = group

    # group used for performing augmentation
    cfg.symmetria.augment.group = group

    # group used for computing symmetry measure
    cfg.symmetria.measure.group = group

    # group used for canonicalization
    cfg.symmetria.canon.group = group
    cfg.symmetria.canon.pretrain.group = group

    # extra atoms for lattice visualization
    cfg.pyvista.extra_atoms = []

    # camera params for lattice visualization
    cfg.pyvista.camera = {
        'position': [10,-20,12],
        'viewup': [0,0,1],
        'zoom': 1.0,
        'extra_lines': [
            np.array([[0.,0.,1.], [1.,0.,1.]]) * L_Bohr,
            np.array([[1.,1.,1.], [1.,0.,1.]]) * L_Bohr,
            np.array([[1.,0.,0.], [1.,0.,1.]]) * L_Bohr,
        ]
    }

    # object params for symmetry visualization
    cfg.pyvista.symmetria.obj_path = 'utils/asymmetric_obj.stl'
    cfg.pyvista.symmetria.rotate_vec = [-1,-1,1]
    cfg.pyvista.symmetria.rotate_angle = 45
    
    obj_lengths = [14., 10.,  6.]
    target_sizes = np.linalg.norm(cell.a, axis=0) / 20
    cfg.pyvista.symmetria.scale = target_sizes / obj_lengths
    cfg.pyvista.symmetria.translate = np.array([.25, .1, .075]) @ cell.a


    return cfg