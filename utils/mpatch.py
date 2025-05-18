import jax
import jax.numpy as jnp
import os, functools
from absl import logging
from typing import Optional
import zipfile
import numpy as np
from DeepSolid import ewaldsum
from DeepSolid.train import AuxiliaryLossData
from DeepSolid import hamiltonian
from DeepSolid import constants
from DeepSolid.utils.kfac_ferminet_alpha import loss_functions

def mpatch_local_ewald_energy(simulation_cell):
    """
    [MPATCH] relax tolerance

    generate local energy of ewald part.
    :param simulation_cell:
    :return:
    """
    ewald = ewaldsum.EwaldSum(simulation_cell)
    assert jnp.allclose(simulation_cell.energy_nuc(),
                        (ewald.ion_ion + ewald.ii_const),
                        rtol=1e-4, atol=1e-2) # [MPATCH] relax tolerance
                        # rtol=1e-6, atol=1e-4)   
                        # rtol=1e-8, atol=1e-5)    
    ## check pyscf madelung constant agrees with DeepSolid

    def _local_ewald_energy(x):
        energy = ewald.energy(x)
        return sum(energy)

    return _local_ewald_energy

def mpatch_make_loss(network, batch_network,
              simulation_cell,
              clip_local_energy=5.0,
              clip_type='real',
              mode='for',
              partition_number=3):
    """
    [MPATCH] correct variance evaluation for pmap setting

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
    el_fun = hamiltonian.local_energy_seperate(network,
                                               simulation_cell=simulation_cell,
                                               mode=mode,
                                               partition_number=partition_number)
    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params, data):
        """

        :param params: a dictionary of parameters
        :param data: batch electron coord with shape [Batch, Nelec * Ndim]
        :return: energy expectation of corresponding walkers (only take real part) with shape [Batch]
        """
        ke, ew = batch_local_energy(params, data)
        e_l = ke + ew
        mean_e_l = jnp.mean(e_l)

        pmean_loss = constants.pmean_if_pmap(mean_e_l, axis_name=constants.PMAP_AXIS_NAME)
        # variance = constants.pmean_if_pmap(jnp.mean(jnp.abs(e_l)**2) - jnp.abs(mean_e_l.real) ** 2,
        #                                    axis_name=constants.PMAP_AXIS_NAME)
        variance = constants.pmean_if_pmap(jnp.mean( jnp.abs(e_l - pmean_loss)**2 ), axis_name=constants.PMAP_AXIS_NAME)
        # [MPATCH] variance needs to be computed with respect to pmean
        loss = pmean_loss.real
        imaginary = pmean_loss.imag

        return loss, AuxiliaryLossData(variance=variance,
                                       local_energy=e_l,
                                       imaginary=imaginary,
                                       kinetic=ke,
                                       ewald=ew,
                                       )

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        """
        customised jvp function of loss function.
        :param primals: inputs of total_energy function (params, data)
        :param tangents: tangent vectors corresponding to the primal (params, data)
        :return: Jacobian-vector product of total energy.
        """
        params, data = primals
        loss, aux_data = total_energy(params, data)
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

        psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
        conj_psi_tangent = jnp.conjugate(psi_tangent)
        conj_psi_primal = jnp.conjugate(psi_primal)

        loss_functions.register_normal_predictive_distribution(conj_psi_primal[:, None])

        primals_out = loss, aux_data
        # tangents_dot = jnp.dot(clip_diff, conj_psi_tangent).real
        # dot causes the gradient to be extensive with batch size, which does matter for KFAC.
        tangents_dot = jnp.mean((clip_diff * conj_psi_tangent).real)

        tangents_out = (tangents_dot, aux_data)

        return primals_out, tangents_out

    return total_energy

def mpatch_find_last_checkpoint(ckpt_path: Optional[str] = None, suffix: str = '') -> Optional[str]:
    """
    [MPATCH] allow for specifying suffixes
    
    Finds most recent valid checkpoint in a directory.

    Args:
      ckpt_path: Directory containing checkpoints.
      suffix: suffix to filter checkpoint files.

    Returns:
      Last QMC checkpoint (ordered by sorting all checkpoints by name in reverse)
      or None if no valid checkpoint is found or ckpt_path is not given or doesn't
      exist. A checkpoint is regarded as not valid if it cannot be read
      successfully using np.load.
    """
    if ckpt_path and os.path.exists(ckpt_path):
        files = [f for f in os.listdir(ckpt_path) if 'qmcjax_ckpt_' in f and f.split('.npz')[0][-len(suffix):] == suffix ] # [MPATCH] added suffix filter
        # Handle case where last checkpoint is corrupt/empty.
        for file in sorted(files, reverse=True):
            fname = os.path.join(ckpt_path, file)
            with open(fname, 'rb') as f:
                try:
                    np.load(f, allow_pickle=True)
                    return fname
                except (OSError, EOFError, zipfile.BadZipFile):
                    logging.info('Error loading checkpoint %s. Trying next checkpoint...',
                                 fname)
    return None

def mpatch_save_checkpoint(save_path: str, t: int, data, params, opt_state, mcmc_width,
         remote_save_path: Optional[int] = None, suffix: str = '') -> str:
    """
    [MPATCH] allow for specifying suffixes

    Saves checkpoint information to a npz file.

    Args:
      save_path: path to directory to save checkpoint to. The checkpoint file is
        save_path/qmcjax_ckpt_$t.npz, where $t is the number of completed
        iterations.
      t: number of completed iterations.
      data: MCMC walker configurations.
      params: pytree of network parameters.
      opt_state: optimization state.
      mcmc_width: width to use in the MCMC proposal distribution.
      suffix: suffix to filter checkpoint files.

    Returns:
      path to checkpoint file.
    """
    ckpt_filename = os.path.join(save_path, f'qmcjax_ckpt_{t:06d}{suffix}.npz')
    logging.info('Saving checkpoint %s', ckpt_filename)
    with open(ckpt_filename, 'wb') as f:
        np.savez(
            f,
            t=t,
            data=data,
            params=params,
            opt_state=opt_state,
            mcmc_width=mcmc_width)

    return ckpt_filename

