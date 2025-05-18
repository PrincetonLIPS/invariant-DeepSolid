'''
    Adaptation of process.py in DeepSolid to support:
    - multi-node training
    - group-averaged network
'''

import jax
import jax.numpy as jnp
import numpy as np
import datetime
import ml_collections
from absl import logging
import time
import chex
import pandas as pd
import functools

from DeepSolid.utils.kfac_ferminet_alpha import optimizer as kfac_optim
from DeepSolid.utils.kfac_ferminet_alpha import utils as kfac_utils

from DeepSolid import constants
from DeepSolid import network
from DeepSolid import train
from DeepSolid import pretrain
from DeepSolid import qmc
from DeepSolid import init_guess
from DeepSolid import hf
from DeepSolid import checkpoint
from DeepSolid.utils import writers
from DeepSolid import estimator

from utils.measure import compute_cohesive_energy, make_symm_measure_computation
from utils.addkeys import make_mcmc_step_with_keys, make_loss_with_keys, pad_data_with_key, split_x_key, pretrain_hartree_fock_with_keys, pretrain_hartree_fock_usingHF_with_keys
from utils.gpave import make_group_ops

def augment_data_if_required(
        data, 
        cfg: ml_collections.ConfigDict, 
        sharded_key = None
    ):
    # no augmentations needed, return the same data
    if not cfg.symmetria.augment.on:
        sharded_key, subkeys = kfac_utils.p_split(sharded_key)
        return data, pad_data_with_key(data, subkeys), sharded_key
    
    # get augmentation operation
    augcfg = cfg.symmetria.augment
    ops_fn = make_group_ops(group=augcfg.group, subsample=augcfg.subsample)
    
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)
    
    aug_data = jax.pmap(
        jax.vmap(lambda x_with_key: ops_fn(*split_x_key(x_with_key)), in_axes=(0,)), 
        axis_name=constants.PMAP_AXIS_NAME
    )(pad_data_with_key(data, subkeys))
    aug_data = aug_data.reshape((aug_data.shape[0], -1, aug_data.shape[-1]))

    # pad aug_data with keys
    sharded_key, subkeys = kfac_utils.p_split(sharded_key)

    return aug_data, pad_data_with_key(aug_data, subkeys), sharded_key
    
def get_params_initialization_key(deterministic):
    '''
    The key point here is to make sure different hosts uses the same RNG key
    to initialize network parameters.
    '''
    if deterministic:
        seed = 888
    else:

        # The overly complicated action here is to make sure different hosts get
        # the same seed.
        @constants.pmap
        def average_seed(seed_array):
            return jax.lax.pmean(jnp.mean(seed_array), axis_name=constants.PMAP_AXIS_NAME)

        local_seed = time.time()
        logging.info('trial')
        a = constants.pmean_if_pmap(local_seed, axis_name=constants.PMAP_AXIS_NAME)
        logging.info(a)
        logging.info('%s Parameter initialization begins.', datetime.datetime.now())
        float_seed = average_seed(jnp.ones(jax.local_device_count()) * local_seed)[0]
        seed = int(1e6 * float_seed)
    
    logging.info('%s Parameter initialization seed ' + str(seed), datetime.datetime.now())
    return jax.random.PRNGKey(seed)

def check_symmetry_mode(cfg: ml_collections.ConfigDict, t: int):
    '''
        Use cfg.symmetria.schedule and t to check what symmetrization to perform

        Return (bool, bool); first bool represents whether to do group-averaging, second bool represents whether to do canonicalization
    '''
    mode = cfg.symmetria.schedule(t)
    if mode == 'OG': 
        return False, False
    elif mode == 'gpave':
        return True, False
    elif mode == 'canon':
        return False, True
    else:
        raise ValueError("Invalid output from cfg.symmetria.schedule. Only 'OG', 'gpave' and 'canon' are supported.")

def process(cfg: ml_collections.ConfigDict, 
            process_id: int,
            get_gradient_for_one_step: bool = False,
            ckpt_restore_filename = False):

    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()

    if cfg.batch_size % num_devices != 0:
        raise ValueError('Batch size must be divisible by number of devices, '
                        f'got batch size {cfg.batch_size} for '
                        f'{num_devices} devices.')
    device_batch_size = cfg.batch_size // num_devices  # batch size per device
    local_batch_size = device_batch_size * num_local_devices  # batch size for all local devices

    data_shape = (num_local_devices, device_batch_size)
    
    ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)
    ckpt_restore_filename = (
            ckpt_restore_filename or
            checkpoint.find_last_checkpoint(ckpt_save_path, suffix=cfg.ckpt_suffix) or
            checkpoint.find_last_checkpoint(ckpt_restore_path, suffix=cfg.ckpt_suffix) )

    simulation_cell = cfg.system.pyscf_cell
    cfg.system.internal_cell = init_guess.pyscf_to_cell(cell=simulation_cell)

    hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
    hartree_fock.init_scf()

    if cfg.system.ndim != 3:
        # The network (at least the input feature construction) and initial MCMC
        # molecule configuration (via system.Atom) assume 3D systems. This can be
        # lifted with a little work.
        raise ValueError('Only 3D systems are currently supported.')

    if cfg.debug.deterministic:
        seed = 666
    else:
        seed = int(1e6 * time.time())

    key = jax.random.PRNGKey(seed)
    # make sure data on each host is initialized differently
    key = jax.random.fold_in(key, jax.process_index())

    system_dict = {
        'klist': hartree_fock.klist,
        'simulation_cell': simulation_cell,
    }
    system_dict.update(cfg.network.detnet)

    # ================================================================================================= #
    #  Network setup:                                                                                   |
    #     - During pre-training, original network is used unless canon.pretrain is True                 |
    #                                                                                                   |
    slater_mat = network.make_solid_fermi_net(**system_dict, method_name='eval_mats')
    slater_logdet = network.make_solid_fermi_net(**system_dict, method_name='eval_logdet')
    slater_slogdet = network.make_solid_fermi_net(**system_dict, method_name='eval_slogdet')
    batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0), out_axes=0)
    batch_slater_mat = jax.vmap(slater_mat.apply, in_axes=(None, 0), out_axes=0)

    if cfg.symmetria.canon.on:
        canon_slater_mat = network.make_canon_net(**system_dict, method_name='eval_mats')
        canon_slater_logdet = network.make_canon_net(**system_dict, method_name='eval_logdet')
        canon_slater_slogdet = network.make_canon_net(**system_dict, method_name='eval_slogdet')
        canon_batch_slater_mat = jax.vmap(canon_slater_mat.apply, in_axes=(None, 0, 0), out_axes=0)
        canon_batch_slater_logdet = jax.vmap(canon_slater_logdet.apply, in_axes=(None, 0, 0), out_axes=0)
        canon_batch_slater_slogdet = jax.vmap(canon_slater_slogdet.apply, in_axes=(None, 0, 0), out_axes=0)
        if cfg.symmetria.canon.pretrain.on:
            canon_pretrain_slater_mat = network.make_pretrain_canon_net(**system_dict, method_name='eval_mats')
            canon_pretrain_slater_logdet = network.make_pretrain_canon_net(**system_dict, method_name='eval_logdet')
            canon_pretrain_slater_slogdet = network.make_pretrain_canon_net(**system_dict, method_name='eval_slogdet')
            canon_pretrain_batch_slater_mat = jax.vmap(canon_pretrain_slater_mat.apply, in_axes=(None, 0, 0), out_axes=0)
            canon_pretrain_batch_slater_logdet = jax.vmap(canon_pretrain_slater_logdet.apply, in_axes=(None, 0, 0), out_axes=0)
            canon_pretrain_batch_slater_slogdet = jax.vmap(canon_pretrain_slater_slogdet.apply, in_axes=(None, 0, 0), out_axes=0)

    # ======= restore / init and pretrain ============================================================ #

    if ckpt_restore_filename:
        t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
            ckpt_restore_filename, local_batch_size)
        logging.info('%s Loaded from training point ' + ckpt_restore_filename, datetime.datetime.now())

    else:
        logging.info('%s No checkpoint found. Training new model.', datetime.datetime.now())
        t_init = 0
        opt_state_ckpt = None
        mcmc_width_ckpt = None
        data = init_guess.init_electrons(key=key, cell=cfg.system.internal_cell,
                                         latvec=simulation_cell.lattice_vectors(),
                                         electrons=simulation_cell.nelec,
                                         batch_size=local_batch_size,
                                         init_width=cfg.mcmc.init_width)
        data = jnp.reshape(data, data_shape + data.shape[1:])
        data = constants.broadcast_all_local_devices(data)
        logging.info('%s Initialized data.', datetime.datetime.now())
        params_initialization_key = get_params_initialization_key(cfg.debug.deterministic)
        params = slater_logdet.init(key=params_initialization_key, data=None)
        logging.info('%s Initialized individual params.', datetime.datetime.now())
        params = constants.replicate_all_local_devices(params)
        logging.info('%s Synced params across devices.', datetime.datetime.now())

    logging.info(
                    f'{datetime.datetime.now():%Y %b %d %H:%M:%S} Initialization: First entry of single-electron stream params: {params["single"][0]["w"][0,0,0]} '
                )

    pmoves = np.zeros(cfg.mcmc.adapt_frequency)
    shared_t = constants.replicate_all_local_devices(jnp.zeros([]))
    shared_mom = kfac_utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_utils.replicate_all_local_devices(
        jnp.asarray(cfg.optim.kfac.damping))
    sharded_key = constants.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = constants.p_split(sharded_key)

    if (t_init == 0 and cfg.pretrain.method == 'net' and
            cfg.pretrain.iterations > 0):
        logging.info('Pretrain using Net distribution.')
        sharded_key, subkeys = constants.p_split(sharded_key)
        if cfg.symmetria.canon.pretrain.on:
            params, data = pretrain_hartree_fock_with_keys(params=params,
                                                        data=data,
                                                        batch_network=canon_pretrain_batch_slater_slogdet,
                                                        batch_orbitals=canon_pretrain_batch_slater_mat,
                                                        sharded_key=subkeys,
                                                        scf_approx=hartree_fock,
                                                        cell=simulation_cell,
                                                        iterations=cfg.pretrain.iterations,
                                                        learning_rate=cfg.pretrain.lr,
                                                        full_det=cfg.network.detnet.full_det,
                                                        )
        else:
            params, data = pretrain.pretrain_hartree_fock(params=params,
                                                        data=data,
                                                        batch_network=batch_slater_slogdet,
                                                        batch_orbitals=batch_slater_mat,
                                                        sharded_key=subkeys,
                                                        scf_approx=hartree_fock,
                                                        cell=simulation_cell,
                                                        iterations=cfg.pretrain.iterations,
                                                        learning_rate=cfg.pretrain.lr,
                                                        full_det=cfg.network.detnet.full_det,
                                                        )

    if (t_init == 0 and cfg.pretrain.method == 'hf' and
            cfg.pretrain.iterations > 0):
        logging.info('Pretrain using Hartree Fock distribution.')
        sharded_key, subkeys = constants.p_split(sharded_key)
        if cfg.symmetria.canon.pretrain.on:
            params, data = pretrain_hartree_fock_usingHF_with_keys(params=params,
                                                                    data=data,
                                                                    batch_orbitals=canon_pretrain_batch_slater_mat,
                                                                    sharded_key=sharded_key,
                                                                    cell=simulation_cell,
                                                                    scf_approx=hartree_fock,
                                                                    iterations=cfg.pretrain.iterations,
                                                                    learning_rate=cfg.pretrain.lr,
                                                                    full_det=cfg.network.detnet.full_det,
                                                                    nsteps=cfg.pretrain.steps,
                                                                    )
        else:
            params, data = pretrain.pretrain_hartree_fock_usingHF(params=params,
                                                                data=data,
                                                                batch_orbitals=batch_slater_mat,
                                                                sharded_key=sharded_key,
                                                                cell=simulation_cell,
                                                                scf_approx=hartree_fock,
                                                                iterations=cfg.pretrain.iterations,
                                                                learning_rate=cfg.pretrain.lr,
                                                                full_det=cfg.network.detnet.full_det,
                                                                nsteps=cfg.pretrain.steps)
    if (t_init == 0 and cfg.pretrain.iterations > 0):
        logging.info('Saving pretrain params')
        checkpoint.save(ckpt_save_path, 0, data, params, None, None, suffix=cfg.ckpt_suffix)

    logging.info(
                    f'{datetime.datetime.now():%Y %b %d %H:%M:%S} Pretrained: First entry of single-electron stream params: {params["single"][0]["w"][0,0,0]} '
                )

    # ================================================================================================= #
    #  Network setup:                                                                                   |
    #     - During training, averaged network may be used depending on                                  |
    #       cfg.symmetria.gpave.on and cfg.symmetria.gpave.freq                                         |
    #                                                                                                   |
    gpave_slater_logdet = network.make_gpave_solid_fermi_net(**system_dict, method_name='eval_logdet')
    gpave_slater_slogdet = network.make_gpave_solid_fermi_net(**system_dict, method_name='eval_slogdet')
    gpave_batch_slater_logdet = jax.vmap(gpave_slater_logdet.apply, in_axes=(None, 0, 0), out_axes=0)
    gpave_batch_slater_slogdet = jax.vmap(gpave_slater_slogdet.apply, in_axes=(None,0,0), out_axes=0)

    # ======= restore / init and pretrain ============================================================ #

    # ======= set up sampling ============================================================ #
    sampling_func = slater_slogdet.apply if cfg.mcmc.importance_sampling else None
    mcmc_step = qmc.make_mcmc_step(batch_slog_network=batch_slater_slogdet,
                                   batch_per_device=device_batch_size,
                                   latvec=jnp.asarray(simulation_cell.lattice_vectors()),
                                   steps=cfg.mcmc.steps,
                                   one_electron_moves=cfg.mcmc.one_electron,
                                   importance_sampling=sampling_func,
                                   )
    gpave_sampling_func = gpave_slater_slogdet.apply if cfg.mcmc.importance_sampling else None
    gpave_mcmc_step = make_mcmc_step_with_keys(batch_slog_network=gpave_batch_slater_slogdet,
                                         batch_per_device=device_batch_size,
                                         latvec=jnp.asarray(simulation_cell.lattice_vectors()),
                                         steps=cfg.mcmc.steps,
                                         one_electron_moves=cfg.mcmc.one_electron,
                                         importance_sampling=gpave_sampling_func,
                                         )
    
    if cfg.symmetria.canon.on:
        canon_sampling_func = canon_slater_slogdet.apply if cfg.mcmc.importance_sampling else None
        canon_mcmc_step = make_mcmc_step_with_keys(batch_slog_network=canon_batch_slater_slogdet,
                                            batch_per_device=device_batch_size,
                                            latvec=jnp.asarray(simulation_cell.lattice_vectors()),
                                            steps=cfg.mcmc.steps,
                                            one_electron_moves=cfg.mcmc.one_electron,
                                            importance_sampling=canon_sampling_func,
                                            )
    # ====== set up loss =========================================================================== #
    total_energy = train.make_loss(network=slater_logdet.apply,
                                   batch_network=batch_slater_logdet,
                                   simulation_cell=simulation_cell,
                                   clip_local_energy=cfg.optim.clip_el,
                                   clip_type=cfg.optim.clip_type,
                                   mode=cfg.optim.laplacian_mode,
                                   partition_number=cfg.optim.partition_number,
                                   )
    if cfg.symmetria.gpave.on:
        gpave_total_energy_with_keys = make_loss_with_keys(network=gpave_slater_logdet.apply,
                                            batch_network=gpave_batch_slater_logdet,
                                            simulation_cell=simulation_cell,
                                            clip_local_energy=cfg.optim.clip_el,
                                            clip_type=cfg.optim.clip_type,
                                            mode=cfg.optim.laplacian_mode,
                                            partition_number=cfg.optim.partition_number,
                                            )
    if cfg.symmetria.canon.on:
        canon_total_energy_with_keys = make_loss_with_keys(network=canon_slater_logdet.apply,
                                                    batch_network=canon_batch_slater_logdet,
                                                    simulation_cell=simulation_cell,
                                                    clip_local_energy=cfg.optim.clip_el,
                                                    clip_type=cfg.optim.clip_type,
                                                    mode=cfg.optim.laplacian_mode,
                                                    partition_number=cfg.optim.partition_number,
                                                    )
    # ====== symmetry logging =========================================================================== #
    if cfg.symmetria.measure.on:
        measure_slater_logdet = network.make_measure_gpave_net(**system_dict, method_name='eval_logdet')

        symmetry_measure = make_symm_measure_computation(measure_f=measure_slater_logdet.apply, 
                                                        f=slater_logdet.apply,
                                                        need_f_key=False,
                                                        )      
        if cfg.symmetria.gpave.on:
            gpave_symmetry_measure = make_symm_measure_computation(measure_f=measure_slater_logdet.apply, 
                                                                f=gpave_slater_logdet.apply,
                                                                need_f_key=True
                                                                )
        if cfg.symmetria.canon.on:
            measure_canon_slater_logdet = network.make_measure_canon_net(**system_dict, method_name='eval_logdet')                     
            canon_symmetry_measure = make_symm_measure_computation(measure_f=measure_canon_slater_logdet.apply, 
                                                                    f=canon_slater_logdet.apply,
                                                                    need_f_key=True,
                                                                    same_key=True # so that the same projection is used
                                                                    )

    # ====== set up optimizer =========================================================================== #
    def learning_rate_schedule(t):
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay)

    val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=True)
    if cfg.symmetria.gpave.on:
        gpave_val_and_grad_with_keys = jax.value_and_grad(gpave_total_energy_with_keys, argnums=0, has_aux=True)
    if cfg.symmetria.canon.on:
        canon_val_and_grad_with_keys = jax.value_and_grad(canon_total_energy_with_keys, argnums=0, has_aux=True)

    if cfg.optim.optimizer == 'kfac':
        # OG optimizer
        optimizer = kfac_optim.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode='fisher_exact',
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME
            # debug=True
        )
        if cfg.symmetria.gpave.on:
            # GPAVE optimizer
            gpave_optimizer = kfac_optim.Optimizer(
                gpave_val_and_grad_with_keys,
                l2_reg=cfg.optim.kfac.l2_reg,
                norm_constraint=cfg.optim.kfac.norm_constraint,
                value_func_has_aux=True,
                learning_rate_schedule=learning_rate_schedule,
                curvature_ema=cfg.optim.kfac.cov_ema_decay,
                inverse_update_period=cfg.optim.kfac.invert_every,
                min_damping=cfg.optim.kfac.min_damping,
                num_burnin_steps=0,
                register_only_generic=cfg.optim.kfac.register_only_generic,
                estimation_mode='fisher_exact',
                multi_device=True,
                pmap_axis_name=constants.PMAP_AXIS_NAME
            )
        if cfg.symmetria.canon.on:
            # canon optimizer
            canon_optimizer = kfac_optim.Optimizer(
                canon_val_and_grad_with_keys,
                l2_reg=cfg.optim.kfac.l2_reg,
                norm_constraint=cfg.optim.kfac.norm_constraint,
                value_func_has_aux=True,
                learning_rate_schedule=learning_rate_schedule,
                curvature_ema=cfg.optim.kfac.cov_ema_decay,
                inverse_update_period=cfg.optim.kfac.invert_every,
                min_damping=cfg.optim.kfac.min_damping,
                num_burnin_steps=0,
                register_only_generic=cfg.optim.kfac.register_only_generic,
                estimation_mode='fisher_exact',
                multi_device=True,
                pmap_axis_name=constants.PMAP_AXIS_NAME
                # debug=True
            )
        ## use one single optimizer state
        sharded_key, subkeys = kfac_utils.p_split(sharded_key)
        gpave_bool, canon_bool = check_symmetry_mode(cfg, t_init)
        # if augmentations are used, need to use augmented data (larger batch size) to initialize optimizer
        processed_data, processed_data_with_keys, sharded_key = augment_data_if_required(data, cfg, sharded_key)
        
        if cfg.symmetria.gpave.on: 
            state = gpave_optimizer.init(params, subkeys, processed_data_with_keys)
            if gpave_bool:
                opt_state = opt_state_ckpt or state  # avoid overwriting ckpted state
        if cfg.symmetria.canon.on: 
            state = canon_optimizer.init(params, subkeys, processed_data_with_keys)
            if canon_bool:
                opt_state = opt_state_ckpt or state  # avoid overwriting ckpted state
        state = optimizer.init(params, subkeys, processed_data)
        if not (gpave_bool or canon_bool):
            opt_state = opt_state_ckpt or state  # avoid overwriting ckpted state 

    elif cfg.optim.optimizer == 'none':
        total_energy = constants.pmap(total_energy)
        if cfg.symmetria.gpave.on:
            gpave_total_energy_with_keys = constants.pmap(gpave_total_energy_with_keys)
        if cfg.symmetria.canon.on:
            canon_total_energy_with_keys = constants.pmap(canon_total_energy_with_keys)
        opt_state = optimizer = gpave_optimizer = canon_optimizer = None
    else:
        raise ValueError('Unrecognized Optimizer.')

    mcmc_step = constants.pmap(mcmc_step)
    if cfg.symmetria.gpave.on:
        gpave_mcmc_step = constants.pmap(gpave_mcmc_step)
    if cfg.symmetria.canon.on:
        canon_mcmc_step = constants.pmap(canon_mcmc_step)

    if mcmc_width_ckpt is not None:
        mcmc_width = constants.broadcast_all_local_devices(jnp.asarray(mcmc_width_ckpt))
    else:
        mcmc_width = constants.replicate_all_local_devices(jnp.asarray(cfg.mcmc.move_width))

    if t_init == 0:
        logging.info('Burning in MCMC chain for %d steps (no averaging used)...', cfg.mcmc.burn_in)
        for t in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = constants.p_split(sharded_key)
            if gpave_bool:
                sharded_key, dkeys = constants.p_split(sharded_key)
                data, pmove = gpave_mcmc_step(params, pad_data_with_key(data, dkeys), subkeys, mcmc_width)
            elif canon_bool:
                sharded_key, dkeys = constants.p_split(sharded_key)
                data, pmove = canon_mcmc_step(params, pad_data_with_key(data, dkeys), subkeys, mcmc_width)
            else:
                data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
        logging.info('Completed burn-in MCMC steps')
        
        sharded_key, dkeys = constants.p_split(sharded_key)
        if gpave_bool:
            init_energy = constants.pmap(gpave_total_energy_with_keys)(params, pad_data_with_key(data, dkeys))[0][0] / simulation_cell.scale
        elif canon_bool:
            init_energy = constants.pmap(canon_total_energy_with_keys)(params, pad_data_with_key(data, dkeys))[0][0] / simulation_cell.scale
        else:
            init_energy = constants.pmap(total_energy)(params, data)[0][0] / simulation_cell.scale
        logging.info('Initial energy for primitive cell: %03.4f E_h',init_energy)

    time_of_last_ckpt = time.time()

    if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
        # If opt_state_ckpt is None, then we're restarting from a previous inference
        # run (most likely due to preemption) and so should continue from the last
        # iteration in the checkpoint. Otherwise, starting an inference run from a
        # training run.
        logging.info('No optimizer provided. Assuming inference run.')
        logging.info('Setting initial iteration to 0.')
        t_init = 0

    train_schema = ['step', 'GPAVE', 'CANON',
                        'energy', 'co_e_eV', 'variance', 'pmove', 'imaginary', 'kinetic', 'ewald', 
                        'symm_ratio_var', 'symm_ratio_mean',
                    ]
    if cfg.log.complex_polarization:
        train_schema.append('complex_polarization')
        polarization = estimator.make_complex_polarization(simulation_cell)
        pmap_polarization = constants.pmap(polarization)
    if cfg.log.structure_factor:
        structure_factor = estimator.make_structure_factor(simulation_cell)
        pmap_structure_factor = constants.pmap(structure_factor)
    
    # use only the main process to write the csv
    if process_id == 0:
        writer_obj = writers.Writer(name=cfg.log.stats_file_name,
                                    schema=train_schema,
                                    directory=ckpt_save_path,
                                    iteration_key=None,
                                    log=False)
        writer = writer_obj.__enter__()

    # TMP
    t_total = cfg.optim.iterations if not get_gradient_for_one_step else t_init + 1

    for t in range(t_init, t_total):
        sharded_key, subkeys = constants.p_split(sharded_key)
        # determine averaging or not
        gpave_bool, canon_bool = check_symmetry_mode(cfg, t)
        if gpave_bool: 
            symm_flag = '[GPAVE]'
            sharded_key, dkeys = constants.p_split(sharded_key)
            current_data = pad_data_with_key(data, dkeys)
            current_mcmc_step = gpave_mcmc_step 
            current_total_energy = gpave_total_energy_with_keys
            current_optimizer = gpave_optimizer
            if cfg.symmetria.measure.on:
                current_symmetry_measure = gpave_symmetry_measure
        elif canon_bool: 
            symm_flag = '[CANON]'
            sharded_key, dkeys = constants.p_split(sharded_key)
            current_data = pad_data_with_key(data, dkeys)
            current_mcmc_step = canon_mcmc_step 
            current_total_energy = canon_total_energy_with_keys
            current_optimizer = canon_optimizer
            if cfg.symmetria.measure.on:
                current_symmetry_measure = canon_symmetry_measure
        else:
            symm_flag = '[OG]'
            current_data = data
            current_mcmc_step = mcmc_step 
            current_total_energy = total_energy
            current_optimizer = optimizer
            if cfg.symmetria.measure.on:
                current_symmetry_measure = symmetry_measure             
            
        # run optimization
        if cfg.optim.optimizer == 'kfac':
            # sampling data for evaluating loss
            timer_st = datetime.datetime.now()
            new_data, pmove = current_mcmc_step(params, current_data, subkeys, mcmc_width)
            timer_ed = datetime.datetime.now()
            logging.info(
                f'{timer_ed:%Y %b %d %H:%M:%S} {symm_flag} Step {t:05d}: Sampling duration {timer_st:%H:%M:%S}-{timer_ed:%H:%M:%S}, '
                + 
                f'{(timer_ed-timer_st).total_seconds()} seconds in total. First entry of data: {new_data[0,0,0]} '
            )

            # augment data if required by config 
            processed_data, processed_data_with_keys, sharded_key = augment_data_if_required(new_data, cfg, sharded_key)

            # optimisation step
            timer_st = datetime.datetime.now()

            # Need this split because MCMC step above used subkeys already
            sharded_key, subkeys = kfac_utils.p_split(sharded_key)
            need_key_for_optim = gpave_bool or canon_bool
            new_params, new_opt_state, new_stats = current_optimizer.step(  # pytype: disable=attribute-error
                params=params,
                state=opt_state,
                rng=subkeys,
                data_iterator=iter([processed_data_with_keys]) if need_key_for_optim else iter([processed_data]),
                momentum=shared_mom,
                damping=shared_damping
            )
                
            timer_ed = datetime.datetime.now()
            logging.info(
                f'{timer_ed:%Y %b %d %H:%M:%S} {symm_flag} Step {t:05d}: Optimization duration {timer_st:%H:%M:%S}-{timer_ed:%H:%M:%S}, '
                + 
                f'{(timer_ed-timer_st).total_seconds()} seconds in total. First entry of single-electron stream params: {new_params["single"][0]["w"][0,0,0]} '
            )
                
            tree = {'params': new_params, 'loss': new_stats['loss'], 'optim': new_opt_state}
            try:
                # We don't do check_nan by default due to efficiency concern.
                # We noticed ~0.2s overhead when performing this nan check
                # at transitional medals.
                if cfg.debug.check_nan:
                    chex.assert_tree_all_finite(tree)
                data = new_data
                params = new_params
                opt_state = new_opt_state
                stats = new_stats
                loss = stats['loss']
                aux_data = stats['aux']
            except AssertionError as e:
                # data, params, opt_state, and stats are not updated
                logging.warn(str(e))
                loss = aux_data = None
        elif cfg.optim.optimizer == 'none':
            data, pmove = current_mcmc_step(params, data, subkeys, mcmc_width)
            if gpave_bool or canon_bool:
                sharded_key, dkey = kfac_utils.p_split(sharded_key)
                loss, aux_data = current_total_energy(params, pad_data_with_key(data, dkey)) 
            else:
                loss, aux_data = current_total_energy(params, data)
        else:
            raise ValueError('Unrecognized Optimizer.')

        shared_t = shared_t + 1
        cell_scale = simulation_cell.scale

        # read stats
        loss_supcell = jnp.mean(loss) if loss is not None else None
        loss = loss_supcell / cell_scale if loss is not None else None
        var = jnp.mean(aux_data.variance) / cell_scale ** 2 if aux_data is not None else None
        imag = jnp.mean(aux_data.imaginary) / cell_scale if aux_data is not None else None
        kinetic = jnp.mean(aux_data.kinetic) / cell_scale if aux_data is not None else None
        ewald = jnp.mean(aux_data.ewald) / cell_scale if aux_data is not None else None
        cohesive_eV = compute_cohesive_energy(loss_supcell, cfg.system.pyscf_cell.atom) / cell_scale if loss is not None else None

        # symmetry logging 
        sharded_key, dkey = kfac_utils.p_split(sharded_key)
        if cfg.symmetria.measure.on:
            symm_ratio_mean, symm_ratio_var = constants.pmap(current_symmetry_measure)(params, pad_data_with_key(data, dkey))
            symm_ratio_var = jnp.mean(symm_ratio_var)
            symm_ratio_mean = jnp.mean(symm_ratio_mean)
        else:
            symm_ratio_var = 'NA'
            symm_ratio_mean = 'NA'

        pmove = jnp.mean(pmove)

        # compile results
        result_dict = {
                        'step': t,
                        'GPAVE': str(gpave_bool),
                        'CANON': str(canon_bool),
                        'energy': np.asarray(loss),
                        'co_e_eV': np.asarray(cohesive_eV),
                        'variance': np.asarray(var),
                        'pmove': np.asarray(pmove),
                        'imaginary': np.asarray(imag),
                        'kinetic': np.asarray(kinetic),
                        'ewald': np.asarray(ewald),
                        'symm_ratio_var': np.asarray(symm_ratio_var), 
                        'symm_ratio_mean': np.asarray(symm_ratio_mean),
                        }

        if cfg.log.complex_polarization:
            polarization_data = pmap_polarization(data)[0]
        if cfg.log.structure_factor:
            structure_factor_data = pmap_structure_factor(data)[0][None, :]
            pd_tabel = pd.DataFrame(structure_factor_data)
            pd_tabel.to_csv(str(ckpt_save_path) + '/structure_factor.csv', mode="a", sep=',', header=False)

        # TMP: if get_gradient_for_one_step, just output the methods for computing gradients and the stats
        if get_gradient_for_one_step is True:
            method_dict = {
                'sharded_key': sharded_key,
                'mcmc_step': functools.partial(current_mcmc_step, width=mcmc_width),
                'old_params': params,
                'old_data': current_data,
                'augment_if_required': functools.partial(augment_data_if_required, cfg=cfg),
                'need_key_for_optim': need_key_for_optim,
                'optimizer_step': functools.partial(current_optimizer.step,
                                    state=opt_state,
                                    momentum=shared_mom,
                                    damping=shared_damping
                )
            }
            return method_dict, result_dict

        # output to csv
        if t % cfg.log.stats_frequency == 0 and loss is not None:
            logging.info(
                f'{datetime.datetime.now():%Y %b %d %H:%M:%S} {symm_flag} Step {t:05d}: {loss:03.4f} E_h, '
                + 
                f'co_E_eV={cohesive_eV:03.4f}, variance={var:03.4f} E_h^2, pmove={pmove:0.2f}, ' 
                +
                f'imaginary part={imag:03.4f}, kinetic={kinetic:.4f} E_h, ewald={ewald:03.4f} E_h, ' 
                +
                (f'symm_ratio_var={symm_ratio_var:.4f}, symm_ratio_mean={symm_ratio_mean:.4f}, ' if cfg.symmetria.measure.on else '')
            )
            if cfg.log.complex_polarization:
                result_dict['complex_polarization'] = np.asarray(polarization_data)
            # use only the main process to write the csv
            if process_id == 0:
                writer.write(t,
                             **result_dict,
                             )
                writer.flush()

        # Update MCMC move width
        if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
            if np.mean(pmoves) > 0.55:
                mcmc_width *= 1.1
            if np.mean(pmoves) < 0.5:
                mcmc_width /= 1.1
            pmoves[:] = 0
        pmoves[t % cfg.mcmc.adapt_frequency] = pmove

        if cfg.log.save_frequency_opt == 'time':
            # save checkpoints according to time
            if (time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60
                or t >= cfg.optim.iterations - 1
                or (cfg.log.save_frequency_in_step > 0 and t % cfg.log.save_frequency_in_step == 0)):
                # no checkpointing in inference mode
                if cfg.optim.optimizer != 'none':
                    checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width, suffix=cfg.ckpt_suffix)

                time_of_last_ckpt = time.time()
        else: 
            # save checkpoints according to iterations
            if t % cfg.log.save_frequency == 0 or t >= cfg.optim.iterations - 1:            
                if cfg.optim.optimizer != 'none':
                    checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width, suffix=cfg.ckpt_suffix)
