import numpy as np
from pyscf.pbc import gto

from DeepSolid import supercell
from DeepSolid.utils import units
from utils.group_utils import get_group

def get_config(base_cfg, cfg_str='C,C,2.4612,1,50,ccpvdz'):
    X, Y, L_Ang, S, z, basis = cfg_str.split(',')
    S = np.diag([int(S), int(S), 1])
    cfg = base_cfg
    L_Ang = float(L_Ang)
    z = float(z)
    L_Bohr = units.angstrom2bohr(L_Ang)

    # Set up cell 
    cell = gto.Cell()
    cell.atom = [[X, [1/2. * L_Bohr, np.cos(np.pi/6)/3 * L_Bohr, 0.]],  
                 [Y, [0, np.cos(np.pi/6)*2/3 * L_Bohr, 0.]]]


    cell.basis = basis
    cell.a = np.array([[1. * L_Bohr, 0., 0.],                         
                       [-0.5 * L_Bohr, np.cos(np.pi/6) * L_Bohr, 0],
                       [0, 0, z]
                       ])
    cell.unit = "B"
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    cfg.system.pyscf_cell = simulation_cell
    cfg.system.pyscf_cell.L_Bohr = L_Bohr
    
    
    # vertices of the asymmetric unit used for orbifold layer (2D for plane group, 3D for space group)
    asu_vertices = np.array([[0., 0., 0.],                         
                             [1/4. * L_Bohr, 1/2. * np.cos(np.pi/6) * L_Bohr, 0.],
                             [1/2. * L_Bohr, 1/2. * np.tan(np.pi/6) * L_Bohr, 0.],
                             [0., 0., z],                         
                             [1/4. * L_Bohr, 1/2. * np.cos(np.pi/6) * L_Bohr, z],
                             [1/2. * L_Bohr, 1/2. * np.tan(np.pi/6) * L_Bohr, z],
                            ])
    # faces
    asu_faces = [
        (0,1,2), (3,4,5), (0,1,3,4), (0,2,3,5), (1,2,4,5)
    ]
    
    # group used for orbifold layer
    group = get_group(
        plane_or_space='plane',
        group_index=17,
        unit_cell=cell.a,
        S=np.diagonal(S),
        L_Bohr=L_Bohr,
        translate=True,
        base_asu_vertices=asu_vertices,
        base_asu_faces=asu_faces,
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
        'position': [0,0,1000],
        'viewup': [0,1,0],
        # 'zoom': 0.5,
    }
    # object params for symmetry visualization
    cfg.pyvista.symmetria.obj_path = 'utils/asymmetric_obj.stl'
    cfg.pyvista.symmetria.rotate_vec = [0,1,0]
    cfg.pyvista.symmetria.rotate_angle = 90
    
    obj_lengths = [14., 10.,  6.]
    target_sizes = np.linalg.norm(cell.a, axis=0) / 8
    target_sizes[2] = 0 # 0 length along z direction
    cfg.pyvista.symmetria.scale = target_sizes / obj_lengths
    cfg.pyvista.symmetria.translate = np.array([target_sizes[0] * 1.5, target_sizes[0] * 1.6, 0])
    return cfg