import numpy as np
from pyscf.pbc import gto

from DeepSolid import supercell
from DeepSolid.utils import units
from utils.group_utils import get_group

def get_config(base_cfg, cfg_str='Li,H,4.02,1,ccpvdz'):
    X, Y, L_Ang, S, basis= cfg_str.split(',')
    S = np.eye(3) * int(S)
    cfg = base_cfg
    L_Ang = float(L_Ang)
    L_Bohr = units.angstrom2bohr(L_Ang)

    # Set up cell
    cell = gto.Cell()
    cell.atom = [[X, [0.0,     0.0,     0.0]],
                 [Y, [0.5 * L_Bohr, 0.5 * L_Bohr, 0.5 * L_Bohr]]]


    cell.basis = basis
    cell.a = (np.ones((3, 3)) - np.eye(3)) * L_Bohr / 2
    cell.unit = "B"
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    cfg.system.pyscf_cell = simulation_cell
    cfg.system.pyscf_cell.L_Bohr = L_Bohr

    # group used for performing averaging
    cfg.symmetria.gpave.translate = False
    cfg.symmetria.gpave.group = get_group(
        plane_or_space='space',
        group_index=225,
        unit_cell=cell.a,
        S=np.diagonal(S),
        L_Bohr=L_Bohr,
        translate=cfg.symmetria.gpave.translate,
    )

    # group used for performing augmentation
    cfg.symmetria.augment.translate = False
    cfg.symmetria.augment.group = get_group(
        plane_or_space='space',
        group_index=225,
        unit_cell=cell.a,
        S=np.diagonal(S),
        L_Bohr=L_Bohr,
        translate=cfg.symmetria.augment.translate,
    )

    # group used for computing symmetry measure
    cfg.symmetria.measure.translate = False
    cfg.symmetria.measure.group = get_group(
        plane_or_space='space',
        group_index=225,
        unit_cell=cell.a,
        S=np.diagonal(S),
        L_Bohr=L_Bohr,
        translate=cfg.symmetria.measure.translate,
    )

    # extra atoms for lattice visualization
    extra_atoms_in_primitive = [
            [X, [0.5 * L_Bohr,     0.5 * L_Bohr,     0.0         ]],
            [X, [0.0,              0.5 * L_Bohr,     0.5 * L_Bohr]],
            [X, [0.5 * L_Bohr,     0.0,              0.5 * L_Bohr]],
            [X, [1.0 * L_Bohr,     0.0,     0.0]],
            [X, [0.0,     1.0 * L_Bohr,     0.0]],
            [X, [1.0 * L_Bohr,     1.0 * L_Bohr,    0.0]],
            [X, [1.0 * L_Bohr,     0.5 * L_Bohr,    0.5 * L_Bohr]],
            [X, [0.5 * L_Bohr,     1.0 * L_Bohr,    0.5 * L_Bohr]],
            [X, [0.5 * L_Bohr,     0.5 * L_Bohr,     1.0 * L_Bohr]],
            [X, [0.0,     0.0,     1.0 * L_Bohr]],
            [X, [1.0 * L_Bohr,     0.0,     1.0 * L_Bohr]],
            [X, [0.0,     1.0 * L_Bohr,     1.0 * L_Bohr]],
            [X, [1.0 * L_Bohr,     1.0 * L_Bohr,    1.0 * L_Bohr]],
            [Y, [0.5 * L_Bohr, 0., 0.]],
            [Y, [0., 0.5 * L_Bohr, 0.]],
            [Y, [0., 0., 0.5 * L_Bohr]],
            [Y, [0.5 * L_Bohr,     1.0 * L_Bohr,    0.0]],
            [Y, [0.5 * L_Bohr,     0.0,             1.0 * L_Bohr]],
            [Y, [1.0 * L_Bohr,     0.5 * L_Bohr,    0.0]],
            [Y, [0.0,              0.5 * L_Bohr,    1.0 * L_Bohr]],
            [Y, [1.0 * L_Bohr,     0.0,             0.5 * L_Bohr]],
            [Y, [0.0,              1.0 * L_Bohr,    0.5 * L_Bohr]],
        ]
    cfg.pyvista.extra_atoms = [
        ( extra_atoms_in_primitive[a][0], np.array(extra_atoms_in_primitive[a][1]) + np.array([i,j,k])*L_Bohr )
        for a,i,j,k in np.ndindex(len(extra_atoms_in_primitive), int(S[0,0]), int(S[1,1]), int(S[2,2]))
    ]
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
