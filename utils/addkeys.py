'''
    Patch network-related functions so that jax key objects are supplied, which allows for random subsampling during the training
'''
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Tuple, Union
import time
from absl import logging
import jax
import jax.numpy as jnp
import functools, optax
from DeepSolid import constants, qmc, train, hamiltonian, hf
from DeepSolid.utils.kfac_ferminet_alpha import loss_functions, optimizer, utils
import numpy as np

'''
    Tools
'''
base_key = jax.random.PRNGKey(int(1e6 * time.time()))
base_key_two = jax.random.PRNGKey(int(2e6 * time.time()))

def split_x_key(x_with_key):
    x = x_with_key[:-1]
    key = x_with_key[-1].astype(jnp.int32)
    return x, key

def split_data_keys(data_with_key_ints):
    data, seeds = jax.vmap(lambda y: (y[:-1], y[-1]), in_axes=(0,))(data_with_key_ints)
    return data, seeds

def pad_data_with_key(data, key):
    # convert key to an array of integer seeds, which has the same first two dimensions as data
    seeds = jax.pmap(lambda k: jax.random.split(k, num=data.shape[1]), axis_name=constants.PMAP_AXIS_NAME)(key)[:,:,0]
    return jnp.concatenate([data, jnp.expand_dims(seeds, axis=-1)], axis=-1)

def pad_data_with_key_single_thread(data, key):
    # pad_data_with_key but no pmap
    seeds = jax.random.split(key, num=data.shape[0])[:,0]
    return jnp.concatenate([data, jnp.expand_dims(seeds, axis=-1)], axis=-1)

'''
    Add keys to mcmc and loss
'''
def make_mcmc_step_with_keys(batch_slog_network,
                   batch_per_device,
                   latvec,
                   steps=10,
                   atoms=None,
                   importance_sampling=None,
                   one_electron_moves=False,
                   ):
    """Creates the MCMC step function. Differs from original DS to allow for network to take in keys

    Args:
      batch_slog_network: function, signature (params, x, key), which evaluates the log of
        the wavefunction (square root of the log probability distribution) at x
        given params. Inputs and outputs are batched.
      batch_per_device: Batch size per device.
      latvec: lattice vector of primitive cell.
      steps: Number of MCMC moves to attempt in a single call to the MCMC step
        function.
      atoms: atom positions. If given, an asymmetric move proposal is used based
        on the harmonic mean of electron-atom distances for each electron.
        Otherwise the (conventional) normal distribution is used.
      importance_sampling: if true, importance sampling is used for MCMC.
      Otherwise, Metropolis method is used.
      one_electron_moves: If true, attempt to move one electron at a time.
        Otherwise, attempt one all-electron move per MCMC step.


    Returns:
      Callable which performs the set of MCMC steps.
    """
    if importance_sampling is not None:
        if one_electron_moves:
            raise ValueError('Importance sampling for one elec move is not implemented yet')
        else:
            logging.info('Using importance sampling')
            func = jax.value_and_grad(importance_sampling, argnums=1)
            inner_fun = qmc.importance_update
    else:
        func = batch_slog_network
        if one_electron_moves:
            logging.info('Using one electron Metropolis sampling')
            inner_fun = qmc.mh_one_electron_update
        else:
            logging.info('Using Metropolis sampling')
            inner_fun = qmc.mh_update

    # @jax.jit
    def mcmc_step(params, data_with_keys, key, width):
        """Performs a set of MCMC steps.

        Args:
          params: parameters to pass to the network.
          data: (batched) MCMC configurations to pass to the network, padded with keys
          key: RNG state.
          width: standard deviation to use in the move proposal.

        Returns:
          (data, pmove), where data is the updated MCMC configurations, key the
          updated RNG state and pmove the average probability a move was accepted.
        """
        
        data, dkeys = split_data_keys(data_with_keys)


        def step_fn_with_key(i, x):
            base_key_i = jax.random.fold_in(base_key, i)
            new_dkeys = jax.vmap(lambda y: jax.random.fold_in(base_key_i, y)[0], in_axes=(0,))(dkeys)
            return inner_fun(params, lambda p, d: func(p, d, new_dkeys), *x,
                            latvec=latvec, stddev=width,
                            atoms=atoms, i=i)


        nelec = (data_with_keys.shape[-1] - 1 ) // 3
        nsteps = nelec * steps if one_electron_moves else steps
        # === [allow key] ===
        logprob = 2. * batch_slog_network(params, data, dkeys)
        
        # === [allow key] ===
        data, key, _, num_accepts = jax.lax.fori_loop(0, nsteps, step_fn_with_key,
                                                        (data, key, logprob, 0.))

        pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
        pmove = constants.pmean_if_pmap(pmove, axis_name=constants.PMAP_AXIS_NAME)
        return data, pmove

    return mcmc_step

def make_loss_with_keys(network, batch_network,
              simulation_cell,
              clip_local_energy=5.0,
              clip_type='real',
              mode='for',
              partition_number=3):
    """
    generates loss function used for wavefunction trains.
    :param network: unbatched logdet function of wavefunction
    :param batch_network: batched logdet function of wavefunction
    :param simulation_cell: pyscf object of simulation cell.
    :param clip_local_energy: clip window width of local energy.
    :param clip_type: specify the clip style. real mode clips the local energy in Cartesion style,
    and complex mode in polar style
    :param mode: specify the evaluation style of local energy.
    'for' mode calculates the laplacian of each electron one by one, which is slow but save GPU memory
    'hessian' mode calculates the laplacian in a highly parallized mode, which is fast but require GPU memory
    'partition' mode calculate the laplacian in a moderate way.
    :param partition_number: Only used if 'partition' mode is employed.
    partition_number must be divisivle by (dim * number of electrons).
    The smaller the faster, but requires more memory.
    :return: the loss function
    """
    el_fun_with_key = local_energy_seperate_with_keys(network,
                                                      simulation_cell=simulation_cell,
                                                      mode=mode,
                                                      partition_number=partition_number)
    batch_local_energy_with_keys = jax.vmap(
                                    lambda params, x_with_key: el_fun_with_key(params, *split_x_key(x_with_key)), 
                                   in_axes=(None, 0))

    @jax.custom_jvp
    def total_energy_with_keys(params, data_and_keys):
        """
        :param params: a dictionary of parameters
        :param data_and_keys: ndarray with shape [Batch, Nelec * Ndim + 1]
        :return: energy expectation of corresponding walkers (only take real part) with shape [Batch]
        """
        ke, ew = batch_local_energy_with_keys(params, data_and_keys)
        e_l = ke + ew
        mean_e_l = jnp.mean(e_l)

        pmean_loss = constants.pmean_if_pmap(mean_e_l, axis_name=constants.PMAP_AXIS_NAME)
        # variance = constants.pmean_if_pmap(jnp.mean(jnp.abs(e_l)**2) - jnp.abs(mean_e_l.real) ** 2,
        #                                    axis_name=constants.PMAP_AXIS_NAME)
        variance = constants.pmean_if_pmap(jnp.mean( jnp.abs(e_l - pmean_loss)**2 ), axis_name=constants.PMAP_AXIS_NAME)
        # [MPATCH] variance needs to be computed with respect to pmean
        loss = pmean_loss.real
        imaginary = pmean_loss.imag

        return loss, train.AuxiliaryLossData(variance=variance,
                                             local_energy=e_l,
                                             imaginary=imaginary,
                                             kinetic=ke,
                                             ewald=ew,
                                            )

    @total_energy_with_keys.defjvp
    def total_energy_jvp_with_keys(primals, tangents):
        """
        customised jvp function of loss function.
        :param primals: inputs of total_energy function (params, data_and_keys)
        :param tangents: tangent vectors corresponding to the primal (params, data_and_keys)
        :return: Jacobian-vector product of total energy.
        """
        params, data_and_keys = primals
        
        loss, aux_data = total_energy_with_keys(params, data_and_keys)
        diff = (aux_data.local_energy - loss)
        if clip_local_energy > 0.0:
            if clip_type == 'complex':
                radius, phase = jnp.abs(diff), jnp.angle(diff)
                radius_tv = constants.pmean_if_pmap(radius.std(), axis_name=constants.PMAP_AXIS_NAME)
                radius_mean = jnp.median(radius)
                radius_mean = constants.pmean_if_pmap(radius_mean, axis_name=constants.PMAP_AXIS_NAME)
                clip_radius = jnp.clip(radius,
                                       radius_mean - radius_tv * clip_local_energy,
                                       radius_mean + radius_tv * clip_local_energy)
                clip_diff = clip_radius * jnp.exp(1j * phase)
            elif clip_type == 'real':
                tv_re = jnp.mean(jnp.abs(diff.real))
                tv_re = constants.pmean_if_pmap(tv_re, axis_name=constants.PMAP_AXIS_NAME)
                tv_im = jnp.mean(jnp.abs(diff.imag))
                tv_im = constants.pmean_if_pmap(tv_im, axis_name=constants.PMAP_AXIS_NAME)
                clip_diff_re = jnp.clip(diff.real,
                                        -clip_local_energy * tv_re,
                                        clip_local_energy * tv_re)
                clip_diff_im = jnp.clip(diff.imag,
                                        -clip_local_energy * tv_im,
                                        clip_local_energy * tv_im)
                clip_diff = clip_diff_re + clip_diff_im * 1j
            else:
                raise ValueError('Unrecognized clip type.')
        else:
            clip_diff = diff

        # take jvp without the key part
        data, keys = split_data_keys(data_and_keys)
        tangent_data, _ = split_data_keys(tangents[1])

        psi_primal, psi_tangent = jax.jvp(lambda params, data: batch_network(params, data, keys), 
                                          (primals[0], data), 
                                          (tangents[0], tangent_data), 
                                        )
        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        loss_functions.register_normal_predictive_distribution(conj_psi_primal[:, None])

        primals_out = loss, aux_data
        # tangents_dot = jnp.dot(clip_diff, conj_psi_tangent).real
        # dot causes the gradient to be extensive with batch size, which does matter for KFAC.
        tangents_dot = jnp.mean((clip_diff * conj_psi_tangent).real)

        tangents_out = (tangents_dot, aux_data)

        return primals_out, tangents_out

    return total_energy_with_keys

'''
    Add keys to kinetic energy
'''

def local_kinetic_energy_with_keys(f):
    '''
    holomorphic mode, which seems dangerous since many op don't support complex number now.
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''
    def _lapl_over_f(params, x, key):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f = jax.grad(f, argnums=1, holomorphic=True)
        grad_f_closure = lambda y: grad_f(params, y, key)

        def _body_fun(i, val):
            primal, tangent = jax.jvp(grad_f_closure, (x + 0j, ), (eye[i] + 0j, ))
            return val + tangent[i] + primal[i] ** 2

        return -0.5 * jax.lax.fori_loop(0, ne, _body_fun, 0.0)

    return _lapl_over_f


def local_kinetic_energy_real_imag_with_keys(f):
    '''
    evaluate real and imaginary part of laplacian.
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''
    def _lapl_over_f(params, x, key):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y, k: f(p, y, k).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y, k: f(p, y, k).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y, key)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y, key)

        def _body_fun(i, val):
            primal_real, tangent_real = jax.jvp(grad_f_real_closure, (x, ), (eye[i], ))
            primal_imag, tangent_imag = jax.jvp(grad_f_imag_closure, (x, ), (eye[i], ))
            kine_real = val[0] + tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2
            kine_imag = val[1] + tangent_imag[i] + 2 * primal_real[i] * primal_imag[i]
            return [kine_real, kine_imag]

        result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip(result, complex)]

    return lambda p, y, k: _lapl_over_f(p, y, k)


def local_kinetic_energy_real_imag_dim_batch_with_keys(f):
    '''
    evaluate real and imaginary part of laplacian, in which vamp is used to accelerate.
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''

    def _lapl_over_f(params, x, key):
        ne = x.shape[-1]
        eye = jnp.eye(ne)
        grad_f_real = jax.grad(lambda p, y, k: f(p, y, k).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y, k: f(p, y, k).imag, argnums=1)
        grad_f_real_closure = lambda y: grad_f_real(params, y)
        grad_f_imag_closure = lambda y: grad_f_imag(params, y)

        def _body_fun(dummy_eye):
            primal_real, tangent_real = jax.jvp(grad_f_real_closure, (x, ), (dummy_eye, ))
            primal_imag, tangent_imag = jax.jvp(grad_f_imag_closure, (x, ), (dummy_eye, ))
            kine_real = ((tangent_real + primal_real ** 2 - primal_imag ** 2) * dummy_eye).sum()
            kine_imag = ((tangent_imag + 2 * primal_real * primal_imag) * dummy_eye).sum()
            return [kine_real, kine_imag]

        # result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
        result = jax.vmap(_body_fun, in_axes=0)(eye)
        result = [re.sum() for re in result]
        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip(result, complex)]

    return lambda p, y, k: _lapl_over_f(p, y, k)


def local_kinetic_energy_real_imag_hessian_with_keys(f):
    '''
    Use jax.hessian to evaluate laplacian, which requires huge amount of memory.
    :param f: function return the logdet of wavefunction
    :return: local kinetic energy
    '''
    def _lapl_over_f(params, x, key):
        ne = x.shape[-1]
        grad_f_real = jax.grad(lambda p, y, k: f(p, y, k).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y, k: f(p, y, k).imag, argnums=1)
        hessian_f_real = jax.hessian(lambda p, y, k: f(p, y, k).real, argnums=1)
        hessian_f_imag = jax.hessian(lambda p, y, k: f(p, y, k).imag, argnums=1)
        v_grad_f_real = grad_f_real(params, x, key)
        v_grad_f_imag = grad_f_imag(params, x, key)
        real_kinetic = jnp.trace(hessian_f_real(params, x, key, ),) + jnp.sum(v_grad_f_real**2) - jnp.sum(v_grad_f_imag**2)
        imag_kinetic = jnp.trace(hessian_f_imag(params, x, key, ),) + jnp.sum(2 * v_grad_f_real * v_grad_f_imag)

        complex = [1., 1j]
        return [-0.5 * re * com for re, com in zip([real_kinetic, imag_kinetic], complex)]

    return lambda p, y, k: _lapl_over_f(p, y, k)


def local_kinetic_energy_partition_with_keys(f, partition_number=3):
  '''
  Try to parallelize the evaluation of laplacian
  :param f: bfunction return the logdet of wavefunction
  :param partition_number: partition_number must be divisivle by (dim * number of electrons).
  The smaller the faster, but requires more memory.
  :return: local kinetic energy
  '''
  vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

  def _lapl_over_f(params, x, key):
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f_real = jax.grad(lambda p, y, k: f(p, y, k).real, argnums=1)
    grad_f_imag = jax.grad(lambda p, y, k: f(p, y, k).imag, argnums=1)
    grad_f_closure_real = lambda y: grad_f_real(params, y)
    grad_f_closure_imag = lambda y: grad_f_imag(params, y)

    eyes = jnp.asarray(jnp.array_split(eye, partition_number))
    def _body_fun(val, e):
        primal_real, tangent_real = vjvp(grad_f_closure_real, (x, ), (e, ))
        primal_imag, tangent_imag = vjvp(grad_f_closure_imag, (x, ), (e, ))
        return val, ([primal_real, primal_imag], [tangent_real, tangent_imag])
    _, (plist, tlist) = \
        jax.lax.scan(_body_fun, None, eyes)
    primal = [primal.reshape((-1, primal.shape[-1])) for primal in plist]
    tangent = [tangent.reshape((-1, tangent.shape[-1])) for tangent in tlist]

    real_kinetic = jnp.trace(tangent[0]) + jnp.trace(primal[0]**2).sum() - jnp.trace(primal[1]**2).sum()
    imag_kinetic = jnp.trace(tangent[1]) + jnp.trace(2 * primal[0] * primal[1]).sum()
    return [-0.5 * real_kinetic, -0.5 * 1j * imag_kinetic]

  return _lapl_over_f

def local_energy_seperate_with_keys(f, simulation_cell, mode='for', partition_number=3):
    """
    genetate the local energy function.
    :param f: function return the logdet of wavefunction.
    :param simulation_cell: pyscf object of simulation cell.
    :param mode: specify the evaluation style of local energy.
    'for' mode calculates the laplacian of each electron one by one, which is slow but save GPU memory
    'hessian' mode calculates the laplacian in a highly parallized mode, which is fast but require GPU memory
    'partition' mode calculate the laplacian in a moderate way.
    :param partition_number: Only used if 'partition' mode is employed.
    partition_number must be divisivle by (dim * number of electrons).
    The smaller the faster, but requires more memory.
    :return: the local energy function.
    """

    if mode == 'for':
        ke_ri = local_kinetic_energy_real_imag_with_keys(f)
    elif mode == 'hessian':
        ke_ri = local_kinetic_energy_real_imag_hessian_with_keys(f)
    elif mode == 'dim_batch':
        ke_ri = local_kinetic_energy_real_imag_dim_batch_with_keys(f)
    elif mode == 'partition':
        ke_ri = local_kinetic_energy_partition_with_keys(f, partition_number=partition_number)
    else:
        raise ValueError('Unrecognized laplacian evaluation mode.')
    ke = lambda p, y, k: sum(ke_ri(p, y, k))
    # ke = local_kinetic_energy(f)
    ew = hamiltonian.local_ewald_energy(simulation_cell)

    def _local_energy(params, x, key):
        kinetic = ke(params, x, key)
        ewald = ew(x)
        return kinetic, ewald

    return _local_energy

'''
    Add keys to HF functions
'''

from DeepSolid.pretrain import _batch_slater_slogdet

def pretrain_hartree_fock_with_keys(params,
                                    data,
                                    batch_network,
                                    batch_orbitals,
                                    sharded_key,
                                    cell,
                                    scf_approx: hf.SCF,
                                    full_det=False,
                                    iterations=1000,
                                    learning_rate=5e-3,
                                    ):
    """
    generates a function used for pretrain, and neural network is used as the target sample.
    :param params: A dictionary of parameters.
    :param data: The input data, a 3N dimensional vector.
    :param batch_network: batched function return the slogdet of wavefunction
    :param batch_orbitals: batched function return the orbital matrix of wavefunction
    :param sharded_key: PRNG key
    :param cell: pyscf object of simulation cell
    :param scf_approx: hf.SCF object in DeepSolid. Used to eval the orbital value of Hartree Fock ansatz.
    :param full_det: If true, the determinants are dense, rather than block-sparse.
     True by default, false is still available for backward compatibility.
     Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
     nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
     if False.
    :param iterations: pretrain iterations
    :param learning_rate: learning rate of pretrain
    =======================
    :return: pretrained parameters and electron positions.
    """

    optimizer = optax.adam(learning_rate)
    opt_state_pt = constants.pmap(optimizer.init)(params)
    leading_shape = data.shape[:-1]

    def make_pretrain_step(batch_orbitals,
                       batch_network,
                       latvec,
                       optimizer,
                       full_det=False,
                       ):
        """
        generate the low-level pretrain function
        :param batch_orbitals: batched function return the orbital matrix of wavefunction
        :param batch_network: batched function return the slogdet of wavefunction
        :param latvec: lattice vector of primitive cell
        :param optimizer: optimizer function
        :return: the low-level pretrain function
        """

        def pretrain_step(data, target, params, state, key):
            """
            One iteration of pretraining to match HF.
            :param data: batched input data, a [batch, 3N] dimensional vector.
            :param target: corresponding HF matrix values.
            :param params: A dictionary of parameters.
            :param state: optimizer state.
            :param key: PRNG key.
            :return: pretrained params, data, state, loss value, slogdet of neural network,
            and number of accepted MCMC moves.
            """

            def loss_fn(x_w_keys, p, target):
                """
                loss function
                :param x_w_keys: batched input data with key, a [batch, 3N+1] dimensional vector.
                :param p: A dictionary of parameters.
                :param target: corresponding HF matrix values.
                :return: value of loss function
                """
                x, keys = split_data_keys(x_w_keys)
                predict = batch_orbitals(p, x, keys)
                if full_det:
                    batch_size = predict[0].shape[0]
                    na = target[0].shape[1]
                    nb = target[1].shape[1]
                    target = [jnp.concatenate(
                        (jnp.concatenate((target[0], jnp.zeros((batch_size, na, nb))), axis=-1),
                        jnp.concatenate((jnp.zeros((batch_size, nb, na)), target[1]), axis=-1)),
                        axis=-2)]
                result = jnp.array([jnp.mean(jnp.abs(tar[:, None, ...] - pre)**2)
                                    for tar, pre in zip(target, predict)]).mean()
                return constants.pmean_if_pmap(result, axis_name=constants.PMAP_AXIS_NAME)

            key, dkey = jax.random.split(key)
            val_and_grad = jax.value_and_grad(loss_fn, argnums=1)            
            loss_val, search_direction = val_and_grad(pad_data_with_key_single_thread(data, dkey), params, target)
            search_direction = constants.pmean_if_pmap(
                search_direction, axis_name=constants.PMAP_AXIS_NAME)
            updates, state = optimizer.update(search_direction, state, params)
            params = optax.apply_updates(params, updates)
            key, dkey = jax.random.split(key)
            dkeys = split_data_keys(pad_data_with_key_single_thread(data, dkey))[1]
            logprob = 2 * batch_network(params, data, dkeys)

            key, dkey = jax.random.split(key)
            dkeys = split_data_keys(pad_data_with_key_single_thread(data, dkey))[1]
            data, key, logprob, num_accepts = qmc.mh_update(params=params,
                                                            f=lambda p, d: batch_network(p, d, dkeys),
                                                            x1=data,
                                                            key=key,
                                                            lp_1=logprob,
                                                            num_accepts=0,
                                                            latvec=latvec)
            return data, params, state, loss_val, logprob, num_accepts

        return pretrain_step


    # UNCHANGED FROM HERE ONWARDS

    pretrain_step = make_pretrain_step(batch_orbitals=batch_orbitals,
                                       batch_network=batch_network,
                                       latvec=cell.lattice_vectors(),
                                       optimizer=optimizer,
                                       full_det=full_det,)
    pretrain_step = constants.pmap(pretrain_step)

    for t in range(iterations):
        target = scf_approx.eval_orb_mat(np.array(data.reshape([-1, cell.nelectron, 3]), dtype=np.float64))
        # PYSCF PBC eval_gto seems only accept float64 array, float32 array will easily cause nan or underflow.
        target = [jnp.array(tar) for tar in target]
        target = [tar.reshape([*leading_shape, ne, ne]) for tar, ne in zip(target, cell.nelec) if ne > 0]

        slogprob_target = [2 * jnp.linalg.slogdet(tar)[1] for tar in target]
        slogprob_target = functools.reduce(lambda x, y: x+y, slogprob_target)
        sharded_key, subkeys = constants.p_split(sharded_key)
        data, params, opt_state_pt, loss, logprob, num_accepts = pretrain_step(
            data, target, params, opt_state_pt, subkeys)
        logging.info('Pretrain iter %05d: Loss=%03.6f, pmove=%0.2f, '
                     'Norm of Net prob=%03.4f, Norm of HF prob=%03.4f', 
                     t, loss[0],
                     jnp.mean(num_accepts) / leading_shape[-1],
                     jnp.mean(logprob),
                     jnp.mean(slogprob_target))

    return params, data


def pretrain_hartree_fock_usingHF_with_keys(params,
                                            data,
                                            batch_orbitals,
                                            sharded_key,
                                            cell,
                                            scf_approx: hf.SCF,
                                            iterations=1000,
                                            learning_rate=5e-3,
                                            nsteps=1,
                                            full_det=False,
                                            ):
    """
    generates a function used for pretrain, and HF ansatz is used as the target sample.
    :param params: A dictionary of parameters.
    :param data: The input data, a 3N dimensional vector.
    :param batch_network: batched function return the slogdet of wavefunction
    :param batch_orbitals: batched function return the orbital matrix of wavefunction
    :param sharded_key: PRNG key
    :param cell: pyscf object of simulation cell
    :param scf_approx: hf.SCF object in DeepSolid. Used to eval the orbital value of Hartree Fock ansatz.
    :param full_det: If true, the determinants are dense, rather than block-sparse.
     True by default, false is still available for backward compatibility.
     Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
     nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
     if False.
    :param iterations: pretrain iterations
    :param learning_rate: learning rate of pretrain
    =======================
    :return: pretrained parameters and electron positions.
    """

    optimizer = optax.adam(learning_rate)
    opt_state_pt = constants.pmap(optimizer.init)(params)
    leading_shape = data.shape[:-1]

    def make_pretrain_step(batch_orbitals,
                           latvec,
                           optimizer,
                           ):
        """
        generate the low-level pretrain function
        :param batch_orbitals: batched function return the orbital matrix of wavefunction
        :param latvec: lattice vector of primitive cell
        :param optimizer: optimizer function
        :return: the low-level pretrain function
        """

        def pretrain_step(data, target, params, state, key):
            """
            One iteration of pretraining to match HF.
            :param data: batched input data, a [batch, 3N] dimensional vector.
            :param target: corresponding HF matrix values.
            :param params: A dictionary of parameters.
            :param state: optimizer state.
            :param key: PRNG key.
            :return: pretrained params, data, state, loss value.
            """

            def loss_fn(x_w_keys, p, target):
                """
                loss function
                :param x_w_keys: batched input data with key, a [batch, 3N+1] dimensional vector.
                :param p: A dictionary of parameters.
                :param target: corresponding HF matrix values.
                :return: value of loss function
                """
                x, keys = split_data_keys(x_w_keys)
                predict = batch_orbitals(p, x, keys)
                if full_det:
                    batch_size = predict[0].shape[0]
                    na = target[0].shape[1]
                    nb = target[1].shape[1]
                    target = [jnp.concatenate(
                        (jnp.concatenate((target[0], jnp.zeros((batch_size, na, nb))), axis=-1),
                         jnp.concatenate((jnp.zeros((batch_size, nb, na)), target[1]), axis=-1)),
                        axis=-2)]
                result = jnp.array([jnp.mean(jnp.abs(tar[:, None, ...] - pre) ** 2)
                                    for tar, pre in zip(target, predict)]).mean()
                return constants.pmean_if_pmap(result, axis_name=constants.PMAP_AXIS_NAME)
            
            key, dkey = jax.random.split(key)
            val_and_grad = jax.value_and_grad(loss_fn, argnums=1)
            loss_val, search_direction = val_and_grad(pad_data_with_key_single_thread(data, dkey), params, target)
            search_direction = constants.pmean_if_pmap(
                search_direction, axis_name=constants.PMAP_AXIS_NAME)
            updates, state = optimizer.update(search_direction, state, params)
            params = optax.apply_updates(params, updates)

            return params, state, loss_val

        return pretrain_step

    pretrain_step = make_pretrain_step(batch_orbitals=batch_orbitals,
                                       latvec=cell.lattice_vectors(),
                                       optimizer=optimizer,)
    pretrain_step = constants.pmap(pretrain_step)

    sharded_key, dkeys = constants.p_split(sharded_key)
    batch_network = _batch_slater_slogdet(scf_approx)
    logprob = 2 * batch_network(None, data, pad_data_with_key_single_thread(data, dkeys)[:,:,-1])

    def step_fn(i, inputs):
        base_key_i = jax.random.fold_in(base_key_two, i)
        dkey = split_data_keys(pad_data_with_key_single_thread(data, base_key_i))[1]
        return qmc.mh_update(params,
                             lambda p, d: batch_network(p, d, dkey),
                             *inputs,
                             latvec=cell.lattice_vectors(),
                             )

    for t in range(iterations):

        for i in range(nsteps):
            sharded_key, subkeys = constants.p_split(sharded_key)
            inputs = (data.reshape([-1, cell.nelectron * 3]),
                      sharded_key[0],
                      logprob,
                      0.)
            data, _,  logprob, num_accepts = step_fn(i, inputs)

        data = data.reshape([*leading_shape, -1])
        target = scf_approx.eval_orb_mat(data.reshape([-1, cell.nelectron, 3]))
        target = [tar.reshape([*leading_shape, ne, ne]) for tar, ne in zip(target, cell.nelec) if ne > 0]

        sharded_key, dkeys = constants.p_split(sharded_key)
        slogprob_net = [2 * jnp.linalg.slogdet(net_mat)[1] for net_mat in constants.pmap(batch_orbitals)(params, data, dkeys)]
        slogprob_net = functools.reduce(lambda x, y: x+y, slogprob_net)

        sharded_key, subkeys = constants.p_split(sharded_key)
        params, opt_state_pt, loss = pretrain_step(data, target, params, opt_state_pt, subkeys)

        logging.info('Pretrain iter %05d: Loss=%03.6f, pmove=%0.2f, '
                     'Norm of Net prob=%03.4f, Norm of HF prob=%03.4f',
                     t, loss[0],
                     jnp.mean(num_accepts) / functools.reduce(lambda x, y: x*y, leading_shape),
                     jnp.mean(slogprob_net),
                     jnp.mean(logprob))

    return params, data
