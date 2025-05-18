import sys
import os
import shutil
import pickle
import datetime, time
from ctypes import cdll
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

sys.path.append( os.getcwd() + '/submodules/DeepSolid')
sys.path.append( os.getcwd() + '/submodules/space-groups')

def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module( module )
    return module

def load_cuda_manually(lib_path):        
    cdll.LoadLibrary(lib_path + 'libcublas.so.11')
    cdll.LoadLibrary(lib_path + 'libcudart.so.11.0')
    cdll.LoadLibrary(lib_path + 'libcublasLt.so.11')
    cdll.LoadLibrary(lib_path + 'libcufft.so.10')
    cdll.LoadLibrary(lib_path + 'libcurand.so.10')
    cdll.LoadLibrary(lib_path + 'libcusolver.so.11')
    cdll.LoadLibrary(lib_path + 'libcusparse.so.11')
    cdll.LoadLibrary(lib_path + 'libcudnn.so.8')
    # do not prelocate memory
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def load_metadata(
        log_dir: str,
        mode: str,
        resume: bool,
        dist: bool = False,
        base_cfg_str: str = 'base_config.py',
        cfg_str: str = None,
        optim_cfg_str: str = None,
        sampling_cfg_str: str = None,
        eval_cfg_str: str = None,
        num_processes: int = 1,
        job_id: int = None,
        process_id: int = 0,
    ):
    #
    #  Store metadata for new training, or retrieve metadata for old training
    #
    utctime = datetime.datetime.utcnow().isoformat()
    
    if resume is False:
        # the main process creates the folder
        if process_id == 0:
            # create log folder if not already exists
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            if os.path.isfile(log_dir + '__metadata.pk'):
                raise FileExistsError('Trying to start new training from a folder "'
                                        + log_dir + ' with existing __metadata_' + mode + '.pk". Consider changing --log-dir for new training '
                                        + ' or setting --resume for resuming old training. Exiting.'
                                    )
            
        base_cfg_dir = 'utils/' 
        cfg_dir = 'config/'
        optim_cfg_dir = 'optim_config/'

        shutil.copy( cfg_dir + cfg_str, log_dir )
        shutil.copy( base_cfg_dir + base_cfg_str, log_dir )
        shutil.copy( optim_cfg_dir + optim_cfg_str, log_dir )
        
        metadata = {
            'base_cfg_str': base_cfg_str,
            'cfg_str': cfg_str,
            'optim_cfg_str': optim_cfg_str,
            'dist': dist,
            'utctime': utctime
        }
        
        if dist is True:
            metadata['num_processes'] = num_processes
            metadata['job_id'] = job_id
            metadata['process_id'] = process_id
            metadata['slurm_file'] = log_dir + '_' + str(job_id) + '_' + str(process_id) + '_slurm.out'
        
        # the main process saves the config
        if process_id == 0:
            with open(log_dir + '__metadata.pk', 'wb+') as f:
                pickle.dump(metadata, f)
    else:
        # if resuming training, no cfg, optim_cfg or dist should be specified; will use existing cfg in the log folder
        if mode == 'train':
            if (cfg_str is not None) or (optim_cfg_str is not None) or (dist is not False):
                raise ValueError('Trying to resume training. No --cfg or --optim_cfg or --dist should be supplied. Exiting.')
        elif mode == 'sample':
            if (cfg_str is not None) or (optim_cfg_str is not None):
                raise ValueError('Trying to start sampling. No --cfg or --optim_cfg should be supplied. Exiting.')

        with open(log_dir + '__metadata.pk', 'rb+') as f:
            metadata = pickle.load(f)

        # get options from metadata
        base_cfg_str = metadata['base_cfg_str']
        cfg_str = metadata['cfg_str']
        optim_cfg_str = metadata['optim_cfg_str']
        if mode == 'train':
            dist = metadata['dist']

        # configuration file handling
        if mode == 'sample':
            # copy sampling config
            sampling_cfg_dir = 'sampling_config/'
            shutil.copy( sampling_cfg_dir + sampling_cfg_str, log_dir )
        elif mode == 'train':
            # use the main process to cache the old metadata
            if process_id == 0:
                if dist is True:
                    # match slurm output naming convention
                    metadata_cache_path = log_dir + '_' + str(metadata['job_id']) + '__metadata_' + mode + '.pk'
                else:
                    metadata_cache_path = log_dir + '_OLD_metadata__' + str(metadata['utctime']) + '.pk'

                shutil.copy(log_dir + '__metadata.pk', metadata_cache_path)   

            # update metadata entries
            metadata['utctime'] = utctime
            if dist is True:
                if num_processes != metadata['num_processes'] and mode == 'train':
                    raise ValueError(['Resuming a training with ' + str(metadata['num_processes']) + 
                                    ' process(es) but initiated ' + str(num_processes) + ' process(es)' ])

                metadata['job_id'] = job_id
                metadata['process_id'] = process_id
                metadata['slurm_file'] = log_dir + '_' + str(job_id) + '_' + str(process_id) + '_slurm.out'

            # use the main process to save the new metadata
            if process_id == 0:
                with open(log_dir + '__metadata.pk', 'wb+') as f:
                    pickle.dump(metadata, f)
        elif mode == 'eval':
            # copy eval config if exists
            if eval_cfg_str is not None:
                eval_cfg_dir = 'eval_config/'
                shutil.copy( eval_cfg_dir + eval_cfg_str, log_dir )
        else:
            raise ValueError('mode must take the value "train", "sample" or "eval".')
    
    return metadata, dist, base_cfg_str, cfg_str, optim_cfg_str

def load_network_from_ckpt(cfg, ckpt_restore_filename: str = None):
    # verbatim from process.process to restore checkpoint                
    from DeepSolid import checkpoint, hf, constants, network, train
    import jax
    import jax.numpy as jnp
    from utils.addkeys import pad_data_with_key, make_loss_with_keys

    # obtain checkpoint
    ckpt_restore_path = checkpoint.get_restore_path(cfg.log.save_path)
    if ckpt_restore_filename is None:
        ckpt_restore_filename = checkpoint.find_last_checkpoint(ckpt_restore_path, suffix=cfg.ckpt_suffix)
    else:
        ckpt_restore_filename = cfg.log.save_path + ckpt_restore_filename
    t, init_data, params, _, mcmc_width = checkpoint.restore(ckpt_restore_filename, shape_check=False) 

    # bootstrap data to obtain a data that matches with the current shape (local devices, batch size)
    seed = int(1e6 * time.time())
    key = jax.random.PRNGKey(seed)
    sharded_key = constants.make_different_rng_key_on_all_devices(key)
    sharded_key, subkeys = constants.p_split(sharded_key)

    init_count = init_data.shape[0] * init_data.shape[1]
    init_data = init_data.reshape([init_count] + list(init_data.shape[2:]))
    bootstrap_index = jax.random.choice(subkeys[0], init_count, (jax.local_device_count(), cfg.batch_size))
    data = init_data[bootstrap_index]

    if cfg.symmetria.gpave.on or cfg.symmetria.canon.on:
        sharded_key, dkeys = constants.p_split(sharded_key)
        data = pad_data_with_key(data, dkeys)

    # set up supercell, initialise HF (to obtain a list of k's), and build batch networks
    simulation_cell = cfg.system.pyscf_cell

    hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
    hartree_fock.init_scf()

    system_dict = {
        'klist': hartree_fock.klist,
        'simulation_cell': simulation_cell,
    }
    system_dict.update(cfg.network.detnet)

    logging.info('Initialised simulation cell.')

    # initialise batch networks
    if cfg.symmetria.gpave.on:
        slater_logdet = network.make_gpave_solid_fermi_net(**system_dict, method_name='eval_logdet')
        slater_slogdet = network.make_gpave_solid_fermi_net(**system_dict, method_name='eval_slogdet')
        batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0, 0), out_axes=0)
        batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0, 0), out_axes=0)
    elif cfg.symmetria.canon.on:
        slater_logdet = network.make_canon_net(**system_dict, method_name='eval_logdet')
        slater_slogdet = network.make_canon_net(**system_dict, method_name='eval_slogdet')
        batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0, 0), out_axes=0)
        batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0, 0), out_axes=0)
    else:
        slater_logdet = network.make_solid_fermi_net(**system_dict, method_name='eval_logdet')
        slater_slogdet = network.make_solid_fermi_net(**system_dict, method_name='eval_slogdet')
        batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0), out_axes=0)
        batch_slater_slogdet = jax.vmap(slater_slogdet.apply, in_axes=(None, 0), out_axes=0)

    logging.info('Initialised network.')

    # make total energy evaluation
    if cfg.symmetria.gpave.on or cfg.symmetria.canon.on:
        make_loss = make_loss_with_keys
    else:
        make_loss = train.make_loss

    total_energy = make_loss(network=slater_logdet.apply,
                             batch_network=batch_slater_logdet,
                             simulation_cell=simulation_cell,
                             clip_local_energy=cfg.optim.clip_el,
                             clip_type=cfg.optim.clip_type,
                             mode=cfg.optim.laplacian_mode,
                             partition_number=cfg.optim.partition_number,
                            )    
    logging.info('Initialised energy evaluation.')

    return t, system_dict, data, params, mcmc_width, sharded_key, simulation_cell, slater_logdet, slater_slogdet, batch_slater_logdet, batch_slater_slogdet, total_energy

def sampling_init(cfg, ckpt_restore_filename: str = None):
    '''
        Args:
            cfg: config file compiled from 'run' function
            ckpt_restore_filename: a specific checkpoint file name to recover
    '''
    from DeepSolid import qmc, constants, network
    import jax 
    import jax.numpy as jnp

    t, system_dict, data, params, mcmc_width, sharded_key, simulation_cell, slater_logdet, slater_slogdet, batch_slater_logdet, batch_slater_slogdet, total_energy = load_network_from_ckpt(cfg, ckpt_restore_filename)

    from utils.addkeys import make_mcmc_step_with_keys

    # make mcmc steps
    if cfg.symmetria.gpave.on or cfg.symmetria.canon.on:
        make_mcmc_step = make_mcmc_step_with_keys
    else:
        make_mcmc_step = qmc.make_mcmc_step

    sampling_func = slater_slogdet.apply if cfg.mcmc.importance_sampling else None
    mcmc_step = make_mcmc_step(batch_slog_network=batch_slater_slogdet,
                               batch_per_device=cfg.batch_size//jax.device_count(),
                               latvec=jnp.asarray(simulation_cell.lattice_vectors()),
                               steps=cfg.mcmc.steps,
                               one_electron_moves=cfg.mcmc.one_electron,
                               importance_sampling=sampling_func,
                              )
    mcmc_step = constants.pmap(mcmc_step)
    mcmc_width = constants.replicate_all_local_devices(jnp.asarray(cfg.mcmc.move_width))

    logging.info('Initialised mcmc step.')
    
    # make symmetry measure evaluation
    if cfg.symmetria.measure.on:
        from utils.measure import make_symm_measure_computation
        if cfg.symmetria.canon.on:
            measure_slater_logdet = network.make_measure_canon_net(**system_dict, method_name='eval_logdet')
        else:
            measure_slater_logdet = network.make_measure_gpave_net(**system_dict, method_name='eval_logdet')
        symmetry_measure = make_symm_measure_computation(measure_f=measure_slater_logdet.apply, 
                                                        f=slater_logdet.apply,
                                                        need_f_key=(cfg.symmetria.gpave.on or cfg.symmetria.canon.on),
                                                        same_key=cfg.symmetria.canon.on
                                                        )       
    else:       
        symmetry_measure = None
    
    logging.info('Initialised symmetry measure evaluation.')

    return data, params, t, mcmc_step, mcmc_width, total_energy, symmetry_measure, ckpt_restore_filename, sharded_key, batch_slater_logdet

def mpatch_load_cfg(
        log_dir: str,
        mode: str,
        libcu_lib_path: str = None,
        resume: bool = False,
        dist: bool = False,
        base_cfg_str: str = 'base_config.py',
        cfg_str: str = None,
        optim_cfg_str: str = None,
        sampling_cfg_str: str = None,
        eval_cfg_str: str = None,
        coord_address: str = None,
        num_processes: int = 1,
        job_id: int = None,
        process_id: int = 0,
        timeout: int = None,
        ckpt_restore_filename: str = None,
        x64: bool = False,
        dist_initialize: bool = False,
    ):
    '''
        Load config for DeepSolid

        Args:
            mode: Must take value in 'train' or 'sample' or 'eval'
            resume: whether to resume training. Only resume==True is supported under sample mode
            dist, base_cfg_str, cfg_str, optim_cfg_str: specify these arguments only if resume==False
        
        Return:
            cfg: ml_collections.ConfigDict
            network: (possibly modified) network module from DeepSolid
    '''
    if mode not in ['train', 'sample', 'eval']:
        raise ValueError("the argument mode must be one of 'train', 'sample' or 'eval'")
    if mode in ['sample', 'eval'] and resume is False:
        raise ValueError("only resume==True is supported under sample / eval mode")
        

    #  fix to accommodate old version of jax in DeepSolid
    if libcu_lib_path is not None:
        load_cuda_manually(libcu_lib_path)
    import jax

    #  enable x64 flag
    if x64:
        from jax.config import config as jax_config
        jax_config.update("jax_enable_x64", True)

    #  load metadata
    metadata, dist, base_cfg_str, cfg_str, optim_cfg_str = load_metadata(
        log_dir=log_dir,
        mode=mode,
        resume=resume,
        dist=dist,
        base_cfg_str=base_cfg_str,
        cfg_str=cfg_str,
        optim_cfg_str=optim_cfg_str,
        sampling_cfg_str=sampling_cfg_str,
        eval_cfg_str=eval_cfg_str,
        num_processes=num_processes,
        job_id=job_id,
        process_id=process_id,
    )

    #  initialise jax.distributed
    from utils_slurm.jax_dist import initialize as initialize_with_timeout
    if dist is True and dist_initialize is False:
        # monkey-patch of jax.distributed to allow for timeout
        jax.distributed.initialize = initialize_with_timeout
        jax.distributed.initialize(coordinator_address=coord_address, 
                                    num_processes=num_processes,
                                    process_id=process_id,
                                    initialization_timeout=timeout)

    logging.info('jax devices: ' + str(jax.devices()))
    logging.info('jax local devices: ' + str(jax.local_devices()))

    #  fail the distributed jobs if we don't have the required number of gpus 
    if dist is True and jax.device_count() != num_processes:
        raise Exception('Device count is ' + str(jax.device_count()) 
                        + ' but ' + str(num_processes) + ' processes needed.')
    
    #  add monkey-patches
    from DeepSolid import hamiltonian, train, checkpoint
    from utils.mpatch import mpatch_local_ewald_energy, mpatch_make_loss, mpatch_find_last_checkpoint, mpatch_save_checkpoint
    hamiltonian.local_ewald_energy = mpatch_local_ewald_energy
    train.make_loss = mpatch_make_loss
    checkpoint.find_last_checkpoint = mpatch_find_last_checkpoint
    checkpoint.save = mpatch_save_checkpoint

    import functools
    from DeepSolid import network
    from utils.gpave import make_gpave_solid_fermi_net_from_group, make_canon_net_from_group, make_canon_ops

    # load cfgs
    base_cfg_module = load_module('base_cfg', log_dir + base_cfg_str)
    base_cfg = base_cfg_module.default()

    cfg_module = load_module('cfg', log_dir + cfg_str)
    cfg = cfg_module.get_config(base_cfg)
    optim_cfg_module = load_module('optim_cfg', log_dir + optim_cfg_str)
    cfg = optim_cfg_module.add_optim_opts(cfg)

    if mode == 'sample':
        sampling_cfg_module = load_module('sampling_cfg', log_dir + sampling_cfg_str)
        cfg = sampling_cfg_module.add_sampling_opts(cfg)
        # stored trained parameters are the same across all processes, so for sampling just take process0
        cfg.ckpt_suffix = f'_process0' 
    elif mode == 'train':
        cfg.ckpt_suffix = f'_process{process_id}'
    elif mode == 'eval' and eval_cfg_str is not None:
        eval_cfg_module = load_module('eval_cfg', log_dir + eval_cfg_str)
        cfg = eval_cfg_module.add_eval_opts(cfg)


    # do NOT use subsampling during evaluation 
    if mode in ['sample', 'eval']:
        assert cfg.symmetria.gpave.subsample.on is False

    cfg.log.save_path = log_dir
    
    # add symmetry related net generation function
    if cfg.symmetria.canon.on:
        network.make_canon_net = functools.partial(make_canon_net_from_group,
                                                    group=cfg.symmetria.canon.group,
                                                    eps=cfg.symmetria.canon.eps,
                                                    rand=cfg.symmetria.canon.rand,
                                                    unit_cell_index=cfg.symmetria.canon.unit_cell_index,
                                                    gpave = False,
                                                   )
        if cfg.symmetria.canon.pretrain.on:
            network.make_pretrain_canon_net = functools.partial(make_canon_net_from_group,
                                                                    group=cfg.symmetria.canon.pretrain.group,
                                                                    eps=cfg.symmetria.canon.pretrain.eps,
                                                                    rand=cfg.symmetria.canon.pretrain.rand,
                                                                    unit_cell_index=cfg.symmetria.canon.pretrain.unit_cell_index,
                                                                    gpave = False,
                                                                    mode=cfg.symmetria.canon.pretrain.mode,
                                                                )
    
    network.make_gpave_solid_fermi_net = functools.partial(make_gpave_solid_fermi_net_from_group, 
                                                                group=cfg.symmetria.gpave.group,
                                                                average_over_phase=cfg.symmetria.gpave.over_phase,
                                                                average_before_det=cfg.symmetria.gpave.before_det,
                                                                subsample=cfg.symmetria.gpave.subsample,
                                                                # sequential=cfg.symmetria.gpave.sequential
                                                                )
    
    if cfg.symmetria.measure.on:
        if cfg.symmetria.canon.on:
            network.make_measure_canon_net = functools.partial(make_canon_net_from_group,
                                                                group=cfg.symmetria.measure.group,
                                                                eps=cfg.symmetria.canon.eps,
                                                                rand=cfg.symmetria.canon.rand,
                                                                unit_cell_index=cfg.symmetria.canon.unit_cell_index,
                                                                gpave=True,
                                                                subsample=cfg.symmetria.measure.subsample,
                                                            )
            
        network.make_measure_gpave_net = functools.partial(make_gpave_solid_fermi_net_from_group, 
                                                                group=cfg.symmetria.measure.group,
                                                                subsample=cfg.symmetria.measure.subsample,
                                                                #  sequential=cfg.symmetria.measure.sequential
        )

    if mode == 'train':
        return cfg
    elif mode == 'sample':
        init_data, params, t, mcmc_step, mcmc_width, total_energy, symmetry_measure, ckpt_restore_filename, sharded_key, batch_logdet = sampling_init(cfg, ckpt_restore_filename)
        return init_data, params, cfg, t, mcmc_step, mcmc_width, total_energy, symmetry_measure, ckpt_restore_filename, sharded_key, batch_logdet
    elif mode == 'eval':
        _, _, _, params, _, sharded_key, _, slater_logdet, slater_slogdet, batch_slater_logdet, batch_slater_slogdet, total_energy = load_network_from_ckpt(cfg, ckpt_restore_filename)
        return params, cfg, sharded_key, slater_logdet, slater_slogdet, batch_slater_logdet, batch_slater_slogdet, total_energy