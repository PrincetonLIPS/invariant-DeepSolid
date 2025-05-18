import os, time, functools
from DeepSolid import constants
from DeepSolid.network import init_solid_fermi_net_params, \
                              slogdet_op, construct_periodic_input_features, eval_phase, \
                              construct_symmetric_features, linear_layer, vmap_linear_layer, \
                              isotropic_envelope, diagonal_envelope, full_envelope, logdet_matmul, eval_func
from space_groups.group import SymmetryGroup
from typing import Tuple, Union, Sequence, Optional, List
from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np

FermiLayers = Tuple[Tuple[int, int], ...]

'''
    Tools
'''

base_key = jax.random.PRNGKey(int(1e6 * time.time()))

def make_group_ops(
      group: SymmetryGroup,
      subsample,
      ):
   '''
      generate functions that perform point group actions and within-supercell-translations on inputs

      :param group: SymmetryGroup object that contains the following attributes:
                    - point_ops: a list of point group operations, each of shape [3,3]
                    - translate_ops: a list of translation operations, each of shape [3,3] and diagonal
                    - basis_inv_T_on3d: operation mapping into the SymmetryGroup basis
                    - basis_T_on3d: operation mapping back from the SymmetryGroup basis
                    - unit_cell: [3,3] vectors describing the unit cell
                    - S: 3d array of integers describing the scaling of the supercell
                    - dims: dimension of the SymmetryGroup
      :param subsample: object with the following attributes:
                        - on: if False, all operations are used. If True, use sub-sampling
                        - num: number of samples to draw. Required if subsample is True
                        - replace: number of samples to draw. Required if subsample is True
      :
      :return: vmapped function for performing group operations
   '''
   sup_cell = jnp.diag(group.S) @ group.unit_cell
   sup_cell_inv = jnp.linalg.inv(sup_cell)
   

   ops = jnp.array([ [group.point_ops[i], group.translate_ops[j]] 
                      for i,j in np.ndindex(group.point_ops.shape[0], group.translate_ops.shape[0])])

   def transform_x(x, point_op, translate_op):
    '''
      :param x: 3d vector 
      :param point_op: point symmetry operation, shape [3,3]
      :param translate_op: translation operation, shape [3,3] and diagonal
      :
      :return: 3d vector, x after transformation
    '''      
    input_x = x @ group.basis_inv_T_on3d
    # For space groups, need to add a constant dimension
    if group.dims == 3:
      input_x = jnp.concatenate([input_x, jnp.ones(1)])
    point_op_x = input_x @ point_op.T
    # Trim off the last dimension and revert to original basis
    if group.dims == 3:
      point_op_x = point_op_x[:3]
    point_op_x = point_op_x @ group.basis_T_on3d
    # translate across different unit cells in the supercell
    translated_x = point_op_x + jnp.diagonal(translate_op)[:3]
    # project back into the supercell
    return jnp.mod(translated_x @ sup_cell_inv, jnp.array([1,1,1])) @ sup_cell

   def ops_fn(xs, key_int):
    '''
      :param xs: 3N array 
      :param key_int: int32
      :
      :return: (num_ops, 3N) array
    '''
    assert len(xs.shape) == 1
    if subsample.on:
      ops_subset = ops[
                    jax.random.choice( jax.random.fold_in(base_key, key_int), ops.shape[0], shape=(subsample.num,), replace=subsample.replace)
                  ]
    else:
      ops_subset = ops
      
    output = jax.vmap(
                 jax.vmap(
                    lambda op, x: transform_x(x, op[0], op[1]),
                    in_axes=(None,0),
                 ),
              in_axes=(0,None),
    )( ops_subset, jnp.reshape(xs, (-1, 3)))

    return output.reshape([-1, xs.shape[0]])

   return ops_fn

'''
    Network
'''

def gpave_logdet_matmul(op_xs: List[Sequence[jnp.ndarray]],
                        w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Group-averaged version.
    Combines determinants and takes dot product with weights in log-domain.

    log-sum-exp trick is used for numerical stability
    ====================================================================================
    Args:
      op_xs: List of FermiNet orbitals in each determinant under each group operation. 
        A list with length num_ops of items: Each item is either of length 1 with shape
        (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
        (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
        determinants are factorised into block-diagonals for each spin channel).
      w: weight of each determinant. If none, a uniform weight is assumed.

    Returns:
      ave_op sum_i w_i D_i in the log domain, 
      where ave_op is the average operator over all group operations,
      w_i is the weight of D_i, the i-th determinant (or product of the i-th determinant 
      in each spin channel, if full_det is not used).
    """
    op_sign_in_list, op_slogdet_list = jax.vmap(
                                            lambda x: jax.vmap(slogdet_op, in_axes=(0,))(x), 
                                            in_axes=(0,)
                                        )(op_xs)
    op_sign_in = jnp.prod(op_sign_in_list,axis=1)
    op_slogdet = jnp.sum(op_slogdet_list,axis=1)
    slogdet_max = jnp.max(op_slogdet)
    # log-sum-exp trick
    op_det = jax.vmap(
                        lambda sign_in, slogdet: sign_in * jnp.exp(slogdet - slogdet_max),
                        in_axes=(0,0) 
                )(op_sign_in, op_slogdet)

    if w is None:
        output = jnp.sum(op_det)
    else:
        output = jnp.sum( jnp.matmul(op_det, w) )

    sign_out = jnp.exp(1j*jnp.angle(output))
    slog_out = jnp.log(jnp.abs(output)) + slogdet_max - jnp.log(len(op_xs))
    return sign_out, slog_out

def solid_fermi_net_orbitals_and_phases(params, x,
                                        simulation_cell=None,
                                        klist=None,
                                        atoms=None,
                                        spins=(None, None),
                                        envelope_type=None,
                                        full_det=False):
    """
    Modify from DeepSolid.network.solid_fermi_net_orbitals to output both orbitals (without phases) and phases
    ====================================================================================
    Forward evaluation of the Solid Neural Network up to the orbitals.
     Args:
       params: A dictionary of parameters, containing fields:
         `single`: a list of dictionaries with params 'w' and 'b', weights for the
           one-electron stream of the network.
         `double`: a list of dictionaries with params 'w' and 'b', weights for the
           two-electron stream of the network.
         `orbital`: a list of two weight matrices, for spin up and spin down (no
           bias is necessary as it only adds a constant to each row, which does
           not change the determinant).
         `dets`: weight on the linear combination of determinants
         `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
           multiplicative envelope.
       x: The input data, a 3N dimensional vector.
       simulation_cell: pyscf object of simulation cell.
       klist: Tuple with occupied k points of the spin up and spin down electrons
       in simulation cell.
       spins: Tuple with number of spin up and spin down electrons.
       envelope_type: a string that specifies kind of envelope. One of:
         `isotropic`: envelope is the same in every direction
       full_det: If true, the determinants are dense, rather than block-sparse.
         True by default, false is still available for backward compatibility.
         Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
         nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
         if False.
     Returns:
       orbitals: One (two matrices if full_det is False) that exchange columns under the
       exchange of inputs
       phases: same shape as orbitals.shape[1:]
       to_env: additional variables that may be needed by the envelope, depending on the envelope type.
     """

    ae_, ee_, r_ae, r_ee = construct_periodic_input_features(x, atoms,
                                                             simulation_cell=simulation_cell,
                                                             )
    ae = jnp.concatenate((r_ae, ae_), axis=2)
    ae = jnp.reshape(ae, [jnp.shape(ae)[0], -1])
    ee = jnp.concatenate((r_ee, ee_), axis=2)

    # which variable do we pass to envelope?
    to_env = r_ae if envelope_type == 'isotropic' else ae_

    if envelope_type == 'isotropic':
        envelope = isotropic_envelope
    elif envelope_type == 'diagonal':
        envelope = diagonal_envelope
    elif envelope_type == 'full':
        envelope = full_envelope

    h_one = ae  # single-electron features
    h_two = ee  # two-electron features
    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
    for i in range(len(params['double'])):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)

        # Execute next layer
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][i]))
        h_two_next = jnp.tanh(vmap_linear_layer(h_two, params['double'][i]['w'],
                                                params['double'][i]['b']))
        h_one = residual(h_one, h_one_next)
        h_two = residual(h_two, h_two_next)
    if len(params['double']) != len(params['single']):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][-1]))
        h_one = residual(h_one, h_one_next)
        h_to_orbitals = h_one
    else:
        h_to_orbitals = construct_symmetric_features(h_one, h_two, spins)
    # Note split creates arrays of size 0 for spin channels without any electrons.
    h_to_orbitals = jnp.split(h_to_orbitals, spins[0:1], axis=0)

    active_spin_channels = [spin for spin in spins if spin > 0]
    orbitals = [linear_layer(h, **p)
                for h, p in zip(h_to_orbitals, params['orbital'])]

    for i, spin in enumerate(active_spin_channels):
        nparams = params['orbital'][i]['w'].shape[-1] // 2
        orbitals[i] = orbitals[i][..., :nparams] + 1j * orbitals[i][..., nparams:]

    if envelope_type in ['isotropic', 'diagonal', 'full']:
        orbitals = [envelope(te, param) * orbital for te, orbital, param in
                    zip(jnp.split(to_env, active_spin_channels[:-1], axis=0),
                        orbitals, params['envelope'])]
    # Reshape into matrices and drop unoccupied spin channels.
    orbitals = jnp.array(
                    [jnp.reshape(orbital, [spin, -1, sum(spins) if full_det else spin])
                            for spin, orbital in zip(active_spin_channels, orbitals) if spin > 0]
                )
    phases = jnp.array(
                eval_phase(x, klist=klist, ndim=3, spins=spins, full_det=full_det)
            )
    orbitals = orbitals.swapaxes(1,2)

    # [CHANGES] Do not multiply orbitals by phases here
    # orbitals = [orb * p[None, :, :] for orb, p in zip(orbitals, phases)]
    if full_det:
        orbitals = jnp.concatenate(orbitals, axis=1)
        orbitals = orbitals.reshape([1] + list(orbitals.shape))
        phases = jnp.concatenate(phases, axis=0)
        phases = phases.reshape([1] + list(phases.shape))
    return orbitals, phases, to_env

def ave_logdet_matmul(op_xs: List[Sequence[jnp.ndarray]],
                        w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Averaged version.
    Combines determinants and takes dot product with weights in log-domain.

    log-sum-exp trick is used for numerical stability
    ====================================================================================
    Args:
      op_xs: List of FermiNet orbitals in each determinant under some list of operations. 
        A list with length num_ops of items: Each item is either of length 1 with shape
        (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
        (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
        determinants are factorised into block-diagonals for each spin channel).
      w: weight of each determinant under each operation. If none, a uniform weight is assumed.

    Returns:
      ave_op sum_i w_i D_i in the log domain, 
      where ave_op is the average operator over all group operations,
      w_i is the weight of D_i, the i-th determinant (or product of the i-th determinant 
      in each spin channel, if full_det is not used).
    """
    op_sign_in_list, op_slogdet_list = jax.vmap(
                                            lambda x: jax.vmap(slogdet_op, in_axes=(0,))(x), 
                                            in_axes=(0,)
                                        )(op_xs)
    op_sign_in = jnp.prod(op_sign_in_list,axis=1)
    op_slogdet = jnp.sum(op_slogdet_list,axis=1)
    slogdet_max = jnp.max(op_slogdet)
    # log-sum-exp trick
    op_det = jax.vmap(
                        lambda sign_in, slogdet: sign_in * jnp.exp(slogdet - slogdet_max),
                        in_axes=(0,0) 
                )(op_sign_in, op_slogdet)

    if w is None:
        output = jnp.sum(op_det)
    else:
        output = jnp.sum(op_det * w)

    sign_out = jnp.exp(1j*jnp.angle(output))
    slog_out = jnp.log(jnp.abs(output)) + slogdet_max - jnp.log(len(op_xs))
    return sign_out, slog_out

'''
   Group-averaging Network
'''

def gpave_eval_func(params, x, key_int,
                    compute_orbitals_phases,
                    method_name='eval_slogdet',
                    group_ops=None, 
                    average_over_phase=True,
                    average_before_det=False,
                    ):
    '''
    Group-averaged version of eval_func in DeepSolid.network
    Only overall wavefunction is averaged out and not individual orbitals
    ================================================================
    generates the wavefunction of simulation cell.
    :param params: parameter dict. Argument to DeepSolid.network.eval_func
    :param x: The input data, a 3N dimensional vector.
    :param compute_orbitals_phases: A function for computing orbitals and phases.
    :param method_name: specify the returned function of wavefunction
    =======[GPAVE parameters]=======
    :param key_int: stores an integer (uint32 object) that will be used as seed for generating key_ints for subsampling -- which may or may not be used depending on group_ops
    :param group_ops: a function that take in electron positions and return transformed electron positions
    :param average_over_phase: whether to average the wavefunction with the phase term               
    :param average_before_det: whether to average orbitals before taking the determinant     
    ================================================================               
    :return: required wavefunction
    '''

    op_x = group_ops(x, key_int)
    
    # Compute orbitals and phases under each group operation
    op_orbitals, op_phases = jax.vmap(
                                    lambda x: compute_orbitals_phases(params, x)[:2], in_axes=(0,) 
                                )(op_x)

    if average_over_phase:
        op_orbitals = jnp.einsum('abcde,abde->abcde', op_orbitals, op_phases)
    else:
        # if we don't need to average over phase, use phase computed on untransformed x
        phase = jax.jit(compute_orbitals_phases)(params, x)[1]
        op_orbitals = jnp.einsum('abcde,bde->abcde', op_orbitals, phase)

    # average the orbitals before passing to determinant function
    if average_before_det:
        orbitals = jnp.mean(op_orbitals, axis=0)
        if method_name == 'eval_slogdet':
            _, result = logdet_matmul(orbitals)
        elif method_name == 'eval_logdet':
            sign, slogdet = logdet_matmul(orbitals)
            result = jnp.log(sign) + slogdet
        elif method_name == 'eval_phase_and_slogdet':
            result = logdet_matmul(orbitals)
        elif method_name == 'eval_mats':
            result = orbitals
        else:
            raise ValueError('Unrecognized method name')
        return result
    
    # if average_before_det == False, average the determinants instead
    if method_name == 'eval_slogdet':
        _, result = ave_logdet_matmul(op_orbitals)
    elif method_name == 'eval_logdet':
        sign, slogdet = ave_logdet_matmul(op_orbitals)
        result = jnp.log(sign) + slogdet
    elif method_name == 'eval_phase_and_slogdet':
        result = ave_logdet_matmul(op_orbitals)
    elif method_name == 'eval_mats':
        # used only during pre-training and compared against HF
        # return the orbitals under identity operation i.e. group averaging not used in pretraining
        result = [op_orbitals[0, :, 0], op_orbitals[0, :, 1]]
    else:
        raise ValueError('Unrecognized method name')
    
    return result

def make_gpave_solid_fermi_net_from_group(
    envelope_type: str = 'full',
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    klist=None,
    simulation_cell=None,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
    method_name='eval_logdet',
    group: SymmetryGroup =None, 
    average_over_phase=True,
    average_before_det=False,
    subsample=None,
):
    '''
    Group-averaged version of make_solid_fermi_net in DeepSolid.network
    Only overall wavefunction is averaged out and not individual orbitals
    ================================================================
    generates the wavefunction of simulation cell.
    :param envelope_type: specify envelope
    :param bias_orbitals: whether to contain bias in the last layer of orbitals
    :param use_last_layer: wheter to use two-electron feature in the last layer
    :param klist: occupied k points from HF
    :param simulation_cell: simulation cell
    :param full_det: specify the mode of wavefunction, spin diagonalized or not.
    :param hidden_dims: specify the dimension of one-electron and two-electron layer
    :param determinants: the number of determinants used
    :param after_determinants: deleted
    :param method_name: specify the returned function
    =======[GPAVE parameters]=======
    :param group: SymmetryGroup object                                                                             
    :param average_over_phase: whether to average the wavefunction with the phase term                              
    :param average_before_det: whether to average orbitals before taking the determinant           
    :param subsample: cfg for subsampling, with the following attributes:
        - on: bool, whether to use subsampling
        - num: int, number of subsamples
        - replace: bool, whether to sample with replacement
    ================================================================            
    :return: a haiku like module which contain init and apply method. init is used to initialize the parameter of
    network and apply method perform the calculation.
    '''
    if method_name not in ['eval_slogdet', 'eval_logdet', 'eval_mats', 'eval_phase_and_slogdet']:
        raise ValueError('Method name is not in class dir.')

    method = namedtuple('method', ['init', 'apply'])
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=simulation_cell.original_cell.atom_coords(),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        after_determinants=after_determinants,
    )
    group_ops = make_group_ops(group, subsample)
    compute_orbitals_phases = functools.partial(solid_fermi_net_orbitals_and_phases, 
                                                klist=klist,
                                                simulation_cell=simulation_cell,
                                                atoms=simulation_cell.original_cell.atom_coords(),
                                                spins=simulation_cell.nelec,
                                                envelope_type=envelope_type,
                                                full_det=full_det)
    network = functools.partial(
        gpave_eval_func,
        compute_orbitals_phases=compute_orbitals_phases,
        method_name=method_name,
        group_ops=group_ops,                                    # [GPAVE]
        average_over_phase=average_over_phase,                  # [GPAVE]
        average_before_det=average_before_det,                  # [GPAVE]
    )
    method.init = init
    method.apply = network
    return method


'''
   Canonicalization
'''

import jax.numpy as jnp 
import time 

base_key = jax.random.PRNGKey(int(1e6 * time.time()))

def make_canon_ops(group,
                   eps: float = 1e-2, 
                   rand: bool = False,
                   unit_cell_index: list = [0,0,0], 
                   ):
    '''
        =======[CANONICALIZATION parameters]=======
        :param group: SymmetryGroup object               
        :param eps: float, determines the size of the boundary region used for smoothing the projection
        :param rand: bool, determines whether proj_index and g should be randomized
        :param unit_cell_index: [int, int, int], determines the default unit cell that e1 should be moved into. Default to [0,0,0]
        ================================
        :return: 
            canon_layer: a function that takes in a 3N vector, representing N electrons, and outputs the following
                - weights_G_eps: (len(group.asu_list),) array of float, indicating the weights associated with influence from different asus
                - ys: 3N vector representing the projected electrons (before any g from G_eps is applied)
            batch_canon_layer: batched version of canon_layer
    '''
    
    sup_cell = jnp.diag(group.S) @ group.unit_cell
    sup_cell_inv = jnp.linalg.inv(sup_cell)
    unit_cell = group.unit_cell
    unit_cell_inv = group.unit_cell_inv
    asu_point_op_inv_T_list = jnp.array([asu['point_op_inv_T'] for asu in group.asu_list])
    asu_shift_list = jnp.array([asu['shift'] for asu in group.asu_list])
    asu_center_list = jnp.array([asu['center'] for asu in group.asu_list])
    asu_normal_list = jnp.array([asu['normal_list'] for asu in group.asu_list])
    asu_indices = jnp.arange(len(group.asu_list))

    def transform_x(x, e1_shift, asu_point_op_inv_T, asu_shift):
        '''
            :param x: N x 3 array of electron positions
            :param e1_shift: shift required to move e1 to the index [0,0,0] unit cell
            
        '''
        x_unit_cell = (x @ unit_cell_inv - e1_shift + asu_shift) @ unit_cell
        x_input = x_unit_cell @ group.basis_inv_T_on3d
        if group.dims == 3:
            x_input = jnp.concatenate([x_input, jnp.ones(1)])
        x_output = x_input @ asu_point_op_inv_T
        if group.dims == 3:
            x_output = x_output[:3]
        x_output = x_output @ group.basis_T_on3d
        # project back into the supercell
        return jnp.mod(x_output @ sup_cell_inv, jnp.array([1,1,1])) @ sup_cell


    stability_eps = 1e-10
    where_trick = lambda x: jnp.where(x > 0., x, 1.)

    @jax.custom_jvp
    def phi(x):
        return jnp.where(x > 0. + stability_eps, jnp.exp(-1/where_trick(x)),  0.)

    @phi.defjvp
    def phi_jvp(primals, tangents):
        x = primals[0]
        x_dot = tangents[0]
        primal_out = phi(x)
        tangent_out = phi_grad(x) * x_dot
        return primal_out, tangent_out

    @jax.custom_jvp
    def phi_grad(x):
        return jnp.where(x > 0. + stability_eps, 1/where_trick(x**2) * jnp.exp(-1/where_trick(x)),  0.)

    @phi_grad.defjvp
    def phi_grad_jvp(primals, tangents):
        x = primals[0]
        x_dot = tangents[0]
        primal_out = phi(x)
        tangent_out = jnp.where(x > 0. + stability_eps, - 2/where_trick(x**3) * jnp.exp(-1/where_trick(x)) + 1/where_trick(x**4) * jnp.exp(-1/where_trick(x)),  0.) * x_dot
        return primal_out, tangent_out

    def smoothed_step(w):
        return phi(w) / (phi(w) + phi(1-w))

    def smoothed_relu(w):
        return w * smoothed_step(w)

    def asu_dist(x, center, normals):
        return jnp.sum(
                    jax.vmap(
                        lambda normal: ( smoothed_relu(jnp.dot(x - center, normal) / (jnp.linalg.norm(normal)**2) - 1) )**2,
                        # lambda normal: ( jax.nn.relu(jnp.dot(x - center, normal) / (jnp.linalg.norm(normal)**2) - 1) )**2,
                        in_axes=(0,)
                    )(normals)
        )    
    

    def canon_layer(elecs, key_int, proj_index):
        '''
        :param elecs: N x 3 array 
        :param key_int: int32
        :parma proj_index: int, determines index w.r.t which projection is done
        :return: 
            - weights: weights associated with each g from the list of asus (if rand is False) or a randomly chosen g (if rand is True)
            - op_xs_list: diagonally tranformed 3N array by each g from the list of asus (if rand is False) or a randomly chosen g (if rand is True)
        '''
        # translate a chosen electron to the (0,0,0)-th unit cell
        e1 = elecs[proj_index]
        e1_shifted = jnp.mod(e1 @ unit_cell_inv, jnp.array([1,1,1])) @ unit_cell
        e1_shift = jax.lax.stop_gradient(jnp.floor_divide(e1 @ unit_cell_inv, jnp.array([1,1,1])))

        # compute the distances of e1_shifted to all asus within and bordering the (0,0,0)-th unit cell
        dists = jax.vmap(
            lambda center, normals: asu_dist(e1_shifted, center, normals),
            in_axes=(0,0),
        )(asu_center_list, asu_normal_list)
        unnorm_weights = jax.vmap(
            lambda dist: smoothed_step((eps - dist) / eps) 
            # lambda dist: jax.nn.sigmoid((eps - dist)/eps) 
        )(dists)
        weights = unnorm_weights / jnp.sum(unnorm_weights)

        op_fn = lambda asu_index: jax.vmap(
                                        lambda x: transform_x(x, e1_shift + jnp.array(unit_cell_index), asu_point_op_inv_T_list[asu_index], asu_shift_list[asu_index]),
                                            # jnp.array(unit_cell_index) is added back to e1_shift s.t. e1 is shifted to the targeted unit cell
                )(elecs).flatten()

        if rand:
            perm = jax.random.permutation(jax.random.fold_in(base_key, key_int), asu_indices)
            static_dists = jax.lax.stop_gradient(dists)
            rand_index = perm[jnp.nonzero(static_dists[perm] <= eps, size=1)[0]]
            renorm_factor = jnp.sum(static_dists <= eps)
            w = renorm_factor * weights[rand_index]
            op_xs = op_fn(rand_index)
            return jnp.expand_dims(w,0), jnp.expand_dims(op_xs,0)
        else:
            op_xs_list = jax.vmap(op_fn)(asu_indices)
            renorm_factor = len(group.asu_list)
            return weights * renorm_factor, op_xs_list

    def canon_layer_over_proj_indices(xs, key_int):
        '''
            :param xs: 3N array 
            :param key_int: int32
            :return: batched outputs of canon_layer
        '''
        elecs = jnp.reshape(xs, (-1, 3))
        elecs_shape = jax.lax.stop_gradient(elecs).shape
        if rand:
            indices = jnp.array([jax.random.choice(jax.random.fold_in(base_key, key_int), elecs_shape[0])])
        else:
            indices = jnp.arange(elecs_shape[0])

        return jax.vmap( lambda index: canon_layer(elecs, key_int, index) )(indices)

    return canon_layer_over_proj_indices

def jax_scan_nocarry(f):
    def scan_f(xs):
        return jax.lax.scan(lambda carry, x: (None, f(x)), None, xs)[1]
    return scan_f

def canon_eval_func(params, x, key_int, 
                    canon_ops,
                    compute_orbitals_phases,  
                    method_name: str,
                    group_ops,
                    gpave: bool,
                    ):
    '''
        generates the wavefunction of simulation cell.
        :param params: parameter dict
        :param x: The input data, a 3N dimensional vector.
        :param compute_orbitals_phases: A function for computing orbitals and phases.
        :param canon_ops: A function for canonicalization
        :param method_name: specify the returned function of wavefunction
        :param group_ops: a function that take in electron positions and return transformed electron positions
        :param gpave: bool, whether to average over group elements
        :return: required wavefunction
    '''
    if gpave is False:
        # Compute orbitals and weights over different projection indices and different group elements considered in canonicalization
        weights, op_xs = canon_ops(x, key_int)

        op_orbitals, op_phases = jax.vmap(
                                    lambda k_x: jax.vmap(
                                        lambda kg_x: compute_orbitals_phases(params, kg_x)[:2], 
                                    )(k_x),
        )(op_xs)

        op_orbitals = jnp.einsum('iabcde,iabde->iabcde', op_orbitals, op_phases)
        op_orbitals = jnp.reshape(op_orbitals, [-1] + list(jax.lax.stop_gradient(op_orbitals).shape[2:]))
        weights = jnp.repeat( 
                    jnp.reshape(weights, (-1, 1)), 
                    repeats=jax.lax.stop_gradient(op_orbitals).shape[2],
                    axis=1
        )
    else:
        # Additionally need to average over groups 
        gp_weights, gp_op_xs = jax.vmap(
            lambda gx: canon_ops(gx, key_int)
        )(group_ops(x, key_int + 1)) # different key supplied to group ops

        op_orbitals, op_phases = jax.vmap(
                                        lambda gp_xs: jax.vmap(
                                            lambda k_x: jax.vmap(
                                                lambda kg_x: compute_orbitals_phases(params, kg_x)[:2], in_axes=(0,) 
                                            )(k_x),
                                    )(gp_xs)
                                )(gp_op_xs) 

        op_orbitals = jnp.einsum('jiabcde,jiabde->jiabcde', op_orbitals, op_phases)
        op_orbitals = jnp.reshape(op_orbitals, [-1] + list(jax.lax.stop_gradient(op_orbitals).shape[3:]))
        weights = jnp.repeat( 
                    jnp.reshape(gp_weights, (-1, 1)), 
                    repeats=jax.lax.stop_gradient(op_orbitals).shape[2],
                    axis=1
        )

    if method_name == 'eval_slogdet':
        _, result = ave_logdet_matmul(op_orbitals, weights)
    elif method_name == 'eval_logdet':
        sign, slogdet = ave_logdet_matmul(op_orbitals, weights)
        result = jnp.log(sign) + slogdet
    elif method_name == 'eval_phase_and_slogdet':
        result = ave_logdet_matmul(op_orbitals, weights)
    elif method_name == 'eval_mats':
        # used only during pre-training and compared against HF; returns orbitals under a random canonicalization
        result = [op_orbitals[0, :, 0], op_orbitals[0, :, 1]]
    else:
        raise ValueError('Unrecognized method name')

    return result

def make_canon_net_from_group(
        envelope_type: str = 'full',
        bias_orbitals: bool = False,
        use_last_layer: bool = False,
        klist=None,
        simulation_cell=None,
        full_det: bool = True,
        hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
        determinants: int = 16,
        after_determinants: Union[int, Tuple[int, ...]] = 1,
        method_name='eval_logdet',
        group: SymmetryGroup = None,
        eps: float = 1e-2, 
        rand: bool = False,
        unit_cell_index: list = [0,0,0],
        gpave: bool = False,
        subsample = None,
    ):
    '''
    generates the wavefunction of simulation cell.
    :param envelope_type: specify envelope
    :param bias_orbitals: whether to contain bias in the last layer of orbitals
    :param use_last_layer: wheter to use two-electron feature in the last layer
    :param klist: occupied k points from HF
    :param simulation_cell: simulation cell
    :param full_det: specify the mode of wavefunction, spin diagonalized or not.
    :param hidden_dims: specify the dimension of one-electron and two-electron layer
    :param determinants: the number of determinants used
    :param after_determinants: deleted
    :param method_name: specify the returned function
    =======[CANONICALIZATION parameters]=======
    :param group: SymmetryGroup object                                                       
    :param eps: float, determines the size of the boundary region used for smoothing the projection
    :param rand: bool, determines whether proj_index and g should be randomized
    :param unit_cell_index: [int, int, int], determines the default unit cell that e1 should be moved into. Default to [0,0,0]
    :
    =======[GPAVE parameters (primarily for measuring symmetry)]=========================
    :param gpave: bool, whether to average over group elements
    :param subsample: cfg for subsampling, used only when gpave is True. Contains the following attributes:
        - on: bool, whether to use subsampling
        - num: int, number of subsamples
        - replace: bool, whether to sample with replacement
    :
    ===========================================
    :return: a haiku like module which contain init and apply method. init is used to initialize the parameter of
    network and apply method perform the calculation.
    '''

    if method_name not in ['eval_slogdet', 'eval_logdet', 'eval_mats', 'eval_phase_and_slogdet']:
        raise ValueError('Method name is not in class dir.')

    method = namedtuple('method', ['init', 'apply'])
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=simulation_cell.original_cell.atom_coords(),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        after_determinants=after_determinants,
    )
    compute_orbitals_phases = functools.partial(solid_fermi_net_orbitals_and_phases, 
                                                klist=klist,
                                                simulation_cell=simulation_cell,
                                                atoms=simulation_cell.original_cell.atom_coords(),
                                                spins=simulation_cell.nelec,
                                                envelope_type=envelope_type,
                                                full_det=full_det
    )
    canon_ops = make_canon_ops(group=group,
                                eps=eps,
                                rand=rand,
                                unit_cell_index=unit_cell_index,
    )
    # group ops for gpave
    group_ops = make_group_ops(group, subsample)
    network = functools.partial(
        canon_eval_func,
        canon_ops=canon_ops,
        compute_orbitals_phases=compute_orbitals_phases,
        method_name=method_name,
        group_ops=group_ops,                          
        gpave=gpave,
    )
    method.init = init
    method.apply = network
    return method
