from space_groups import SpaceGroup, PlaneGroup
from space_groups.group import SymmetryGroup
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def get_group(
        plane_or_space: str, 
        group_index: int, 
        unit_cell: np.ndarray, 
        S: np.ndarray,
        L_Bohr: float,
        translate: bool = False,
        base_asu_vertices: np.ndarray = None,
        base_asu_faces: list = None,
    ) -> SymmetryGroup:
    '''
      get SymmetryGroup object with basis_inv_T pre-computed 
      :param plane_or_space: string, either 'plane' or 'space'
      :param group_index: integer
      :param unit_cell: [3,3] vectors describing the unit cell
      :param S: 3d array of integers describing the scaling of the supercell
      :param L_Bohr: float describing the unit length
      :param translate: whether to use translations within the supercell
      :param base_asu_vertices: vectors describing the asymmetric unit chosen as the base
      :param base_asu_faces: list of tuples describing the faces of the asymmetric unit
      :
      :return: SymmetryGroup object
    '''
    if plane_or_space == 'space':
        group = SpaceGroup(group_index)
        group.basis_inv_T_on3d = np.linalg.inv(group.basic_basis).T / L_Bohr
        group.basis_T_on3d = group.basic_basis.T * L_Bohr
    elif plane_or_space == 'plane':
        group = PlaneGroup(group_index)
        # for 2d group, pad basis inverse to 3x3 matrix so that it works on 3d coordinates
        group.basis_inv_T_on3d = np.pad(np.linalg.inv(group.basic_basis).T / L_Bohr, [(0, 1), (0, 1)], mode='constant')
        group.basis_inv_T_on3d[2,2] = 1
        group.basis_T_on3d = np.pad(group.basic_basis.T * L_Bohr, [(0, 1), (0, 1)], mode='constant')
        group.basis_T_on3d[2,2] = 1
    else:
        raise ValueError('The argument "plane_or_space" can only be "plane" or "space".')
    
    # make operations
    group.unit_cell = unit_cell
    group.unit_cell_inv = np.linalg.inv(group.unit_cell)
    group.S = S
    #   get operations associated with point group symmetries
    group.point_ops = np.array([op.to_doubles() for op in group.operations])
    #   add translations, which matters for supercell greater than 1x1x1
    if translate:
      group.translate_ops = np.array([
        np.diag(
           np.concatenate([indices @ unit_cell, [0.]*(group.dims-2) ])
        ) 
        for indices in np.ndindex(int(S[0]),int(S[1]),int(S[2]))
      ])
    else:
       group.translate_ops = np.zeros([1,group.dims+1,group.dims+1])

    
  
    if base_asu_vertices is not None and base_asu_faces is not None:
      # make a list of asus in the unit cell
      base_asu_list = []
      unit_cell_center = np.sum(unit_cell, axis=0) / 2
      jitter = 1e-5

      for point_op in group.point_ops: # assumed the first op is identity
        # obtain the transformed asu and record the transformation that maps it back to the base_asu
        input_asu = base_asu_vertices @ group.basis_inv_T_on3d
        if group.dims == 3:
            input_asu = np.pad(input_asu, [(0,0),(0,1)], mode='constant', constant_values=1)
        asu_before_shift = input_asu @ point_op.T
        if group.dims == 3:
            asu_before_shift = asu_before_shift[:,:3]
        asu_before_shift = asu_before_shift @ group.basis_T_on3d
        # find the point furthest from the unit cell center and shift the asu back into the unit cell if necessary
        max_i = np.argmax(np.linalg.norm(asu_before_shift - unit_cell_center, axis=1))
        possible_shifts = np.array([
          np.floor_divide( (asu_before_shift[max_i] + np.array([ei, ej, ek])) @ group.unit_cell_inv, np.array([1,1,1]))
          for ei in [+jitter, 0, -jitter] 
          for ej in [+jitter, 0, -jitter] 
          for ek in [+jitter, 0, -jitter] 
        ])
        shift = possible_shifts[np.argmin(np.linalg.norm(possible_shifts, axis=1))]
        asu = (asu_before_shift @ group.unit_cell_inv - shift) @ unit_cell

        # compute center of asu and the normal vectors to all faces that start from the center
        center = np.average(asu, axis=0)
        normal_list = []
        for face in base_asu_faces:
          assert len(face) >= 3
          A, B, C = [asu[face[i]] - center for i in range(3)]
          unit_normal = np.cross(B-A, C-A)
          unit_normal = unit_normal / np.linalg.norm(unit_normal)
          normal = unit_normal * np.dot(A, unit_normal)
          normal_list.append(normal)

        base_asu_list.append({
          'point_op_inv_T': np.linalg.inv(point_op.T), 
          'shift': shift,
          'vertices': asu,
          'face_list': base_asu_faces,
          'center': center,
          'normal_list': normal_list,
        })
      group.base_asu_list = base_asu_list
      group.asu_list = [asu for asu in base_asu_list]

      # include the asus from nearby cells that share a boundary with the current cell
      unit_translate_ops = np.array([
          np.concatenate([np.array([i,j,k]) @ unit_cell, [0]*(group.dims-2) ])
          for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]
      ])
      
      for t_op in unit_translate_ops:
         if np.all(t_op == 0): # skip if no translation
            continue
         for asu in base_asu_list:
            # check if the asu shifted by t_op borders the primitive cell
            border_check = np.any([
              (np.all(np.floor_divide((v + t_op + np.array([ei, ej, ek])) @ group.unit_cell_inv, np.array([1,1,1])) == 0)) 
              for ei in [+jitter, 0, -jitter] 
              for ej in [+jitter, 0, -jitter] 
              for ek in [+jitter, 0, -jitter] 
              for v in asu['vertices']   
            ])
            if border_check: # at least one vertex borders the primitive cell
              group.asu_list.append({
                'point_op_inv_T': asu['point_op_inv_T'], 
                'shift': asu['shift'] + t_op,
                'vertices': asu['vertices'] + t_op,
                'face_list': asu['face_list'],
                'center': asu['center'] + t_op,
                'normal_list': asu['normal_list'],
              })
            


    return group