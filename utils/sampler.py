import sys, os, time, pickle, shutil
import numpy as np
from utils.loader import mpatch_load_cfg
import datetime

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

sys.path.append( os.getcwd() + '/submodules/DeepSolid')
sys.path.append( os.getcwd() + '/submodules/space-groups')

class DeepSolidSampler():
    def __init__(
            self,
            log_dir: str,
            sampling_cfg_str: str,
            libcu_lib_path: str = None,
            dist: bool = False,
            coord_address: str = None,
            num_processes: int = 1,
            job_id: int = None,
            process_id: int = 0,
            timeout: int = None,
            ckpt_restore_filename: str = None,
            x64: bool = False,
            symm_measure_only: bool = False,
            stats_prefix: str = '',
        ):
        '''
            config and whether to run in distributed mode are predetermined by the checkpoints saved in log_dir.
        '''
        self.log_dir = log_dir
        self.sampling_cfg_str = sampling_cfg_str
        self.init_data, self.params, self.cfg, self.t, self.mcmc_step, self.mcmc_width, self.total_energy, self.symmetry_measure, self.ckpt_restore_filename, self.sharded_key, self.batch_logdet = \
            mpatch_load_cfg(
                log_dir=log_dir,
                sampling_cfg_str=sampling_cfg_str,
                mode='sample',
                libcu_lib_path=libcu_lib_path,
                dist=dist,
                resume=True,
                coord_address=coord_address,
                num_processes=num_processes,
                job_id=job_id,
                process_id=process_id,
                timeout=timeout,
                ckpt_restore_filename=ckpt_restore_filename,
                x64=x64
            )
        self.batch_size = self.cfg.batch_size
        self.nelec = np.sum(self.cfg.system.pyscf_cell.nelec)
        self.symm_measure_only = symm_measure_only

        self.ckpt_id = self.ckpt_restore_filename.split('.npz')[0].split('ckpt_')[1]

        get_savepath = lambda i, prefix: self.log_dir + prefix + '_' + self.ckpt_id + '__' + self.sampling_cfg_str.split('.py')[0] + \
                                         ('_process' + str(i) if i is not None else '') + '.pk'
        # if coord_address is None:
        #     self.samplespath = get_savepath(None, 'samples')
        #     self.samplespath_all = [ self.samplespath ]
        # else:
        self.samplespath = get_savepath(process_id, 'samples')
        self.samplespath_all = [ get_savepath(i, 'samples') for i in range(num_processes) ]
        
        self.statspath = get_savepath(None, stats_prefix + 'stats' +  ('_symmonly' if symm_measure_only else '' ))
        self.stats_list = []

        self.samples = None # the samples held per process

        self.process_id = process_id
        self.num_processes = num_processes
        
        # important to import this after init
        from DeepSolid import constants
        self.constants = constants

        logging.info('Sampler initialized.')

    '''
        Sampling
    '''
    def load_samples(self):
        if os.path.isfile(self.samplespath):
            with open(self.samplespath, 'rb') as file:
                samples = pickle.load(file) 
            assert samples.shape[1:] == (self.nelec*3,)
            self.samples = samples
            logging.info( f'{datetime.datetime.now():%Y %b %d %H:%M:%S} {self.samples.shape[0]} existing samples found for process {self.process_id}.')
        else:
            logging.info(  f'{datetime.datetime.now():%Y %b %d %H:%M:%S} No existing samples found for process {self.process_id}.')

    def sample_once(self):
        self.sharded_key, subkeys = self.constants.p_split(self.sharded_key)
        data, _ = self.mcmc_step(self.params, self.init_data, subkeys, self.mcmc_width)
        data = data.reshape([-1, data.shape[-1]]) # collapse num_local_devices and device_batch_size dimensions
        return data

    def save_samples(self, samples_list):
        import jax 
        self.samples = jax.numpy.concatenate(samples_list, axis=0)
        with open(self.samplespath, 'wb+') as file:
            pickle.dump(self.samples, file) 

    def draw_samples(
            self,
            required_samples: int,
            save_freq: int = 10,
        ):
        '''
            Draw samples
        '''
        # check existing samples
        if self.samples is None:
            samples_list = []
            existing_samples = 0
        elif self.samples.shape[0] >= required_samples:
            logging.info(  f'{datetime.datetime.now():%Y %b %d %H:%M:%S} Existing samples are more than the required number of samples, {required_samples}. Exiting.')
            return
        else:
            samples_list = [ self.samples ]
            existing_samples = self.samples.shape[0]

        logging.info("====================\n Drawing samples... Saving to " + self.samplespath + "...\n====================")
        num_batches = (required_samples - existing_samples - 1) // self.batch_size + 1
        
        for b in range(num_batches):
            logging.info( f'{datetime.datetime.now():%Y %b %d %H:%M:%S} batch no. {b+1} / {num_batches} of {self.batch_size} samples')
            data = self.sample_once()
            samples_list.append(data)
            if (b+1) % save_freq == 0:
                self.save_samples(samples_list)
            
        self.save_samples(samples_list)
        logging.info("====================\n Sampling completed. \n====================")

    def get_all_samples(self):
        # important to import jax only after init
        import jax
        
        # load samples across all processes
        samples_list = []
        for path in self.samplespath_all:
            if os.path.isfile(path):
                with open(path, 'rb') as file:
                    samples = pickle.load(file) 
                assert samples.shape[1:] == (self.nelec*3,)
                samples_list.append(samples)
        
        if len(samples_list) > 0:
            samples_all = jax.numpy.concatenate(samples_list, axis=0)
            logging.info( str(samples_all.shape[0]) + ' existing samples found.')
        else:
            samples_all = None
            logging.info( 'No existing samples found.')
            
        return samples_all

    def load_all_samples(self):
        samples_all = self.get_all_samples()
        self.samples = samples_all

    '''
        Computing statistics
    '''
    def cache_old_stats(self):
        if os.path.isfile(self.statspath):
            shutil.copy( self.statspath, self.statspath + '_CACHE' )

    def save_stats(self):
        # only used to main process to save 
        if self.process_id == 0:
            with open(self.statspath, 'wb+') as file:
                pickle.dump(self.stats_list, file) 

    def compute_stats(
            self, 
            n_for_each_est: int = 1, # remove this argument?
            save_freq: int = 10
        ):
        '''
            This function assumes each process has the same (number of samples) / (number of devices).
            n_for_each_est is the number of samples used in each empirical estimator of energy and variance
        '''
        import jax
        import jax.numpy as jnp
        from utils.measure import compute_cohesive_energy
        from utils.addkeys import pad_data_with_key
        
        num_devices = jax.device_count()
        num_local_devices = jax.local_device_count()
        if self.samples is None:
            logging.info('No samples found. Exiting.')
            return 
        #assert n_for_each_est > 1
        assert n_for_each_est % num_devices == 0
        
        n_per_device = n_for_each_est // num_devices
        n_this_process = n_per_device * num_local_devices
        num_est = self.samples.shape[0] // n_this_process
        cell_scale = self.cfg.system.pyscf_cell.scale
        self.cache_old_stats()
        self.stats_list = []

        logging.info("====================\n Computing " + str(num_est) + ' estimators across ' + str(num_devices) + 
                " process(es)... \n====================" )

        for i in range(num_est):
            logging.info(  f'{datetime.datetime.now():%Y %b %d %H:%M:%S} Computing estimator {i+1} / {num_est} ...')
            st = i * n_this_process
            ed = (i+1) * n_this_process
            data = self.samples[st:ed, ].reshape([num_local_devices, n_per_device, self.nelec*3])
            if self.symm_measure_only is False:
                if self.cfg.symmetria.gpave.on or self.cfg.symmetria.orbifold.on or (self.cfg.symmetria.canon.on and self.cfg.symmetria.canon.mask is False):
                    self.sharded_key, dkey = self.constants.p_split(self.sharded_key)
                    loss, aux_data = self.constants.pmap(self.total_energy)(self.params, pad_data_with_key(data, dkey))
                else:
                    loss, aux_data = self.constants.pmap(self.total_energy)(self.params, data)

                loss_supcell = jnp.mean(loss)
                loss = loss_supcell / cell_scale
                var = jnp.mean(aux_data.variance) / cell_scale ** 2
                imag = jnp.mean(aux_data.imaginary) / cell_scale
                kinetic = jnp.mean(aux_data.kinetic) / cell_scale
                ewald = jnp.mean(aux_data.ewald) / cell_scale
                cohesive_eV = compute_cohesive_energy(loss_supcell, self.cfg.system.pyscf_cell.atom) / cell_scale
            else:
                loss = None
                var = None
                imag = None
                kinetic = None
                ewald = None
                cohesive_eV = None
            
            
            self.sharded_key, dkey = self.constants.p_split(self.sharded_key)
            if self.cfg.symmetria.measure.on is True or self.symm_measure_only is True:
                symm_ratio_mean, symm_ratio_var = self.constants.pmap(self.symmetry_measure)(self.params, pad_data_with_key(data, dkey))
                symm_ratio_var = jnp.mean(symm_ratio_var)
                symm_ratio_mean = jnp.mean(symm_ratio_mean)
            else:
                symm_ratio_mean = None
                symm_ratio_var = None

            # reported stats are computed per unit cell
            self.stats_list.append({
                'mean': loss,
                'variance': var,
                'imaginary': imag,
                'kinetic': kinetic,
                'ewald': ewald,
                'cohesive_eV': cohesive_eV,
                'symm_ratio_var': symm_ratio_var,
                'symm_ratio_mean': symm_ratio_mean,
                'n_for_each_est': n_for_each_est,
            })
            if (i+1) % save_freq == 0:
                self.save_stats()
        
        self.save_stats()

    def load_stats(self):
        if os.path.isfile(self.statspath):
            with open(self.statspath, 'rb') as file:
                self.stats_list = pickle.load(file) 
        logging.info( str(len(self.stats_list)) + ' existing stats found.' )
   


def get_all_samples(samplespath_all: list):
    '''
       samplespath_all: list of paths to samples
    '''
    # important to import jax only after init
    import jax
    
    # load samples across all processes
    samples_list = []
    for path in samplespath_all:
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                samples = pickle.load(file) 
            samples_list.append(samples)
    
    if len(samples_list) > 0:
        samples_all = jax.numpy.concatenate(samples_list, axis=0)
        logging.info( str(samples_all.shape[0]) + ' existing samples found.')
    else:
        samples_all = None
        logging.info( 'No existing samples found.')
        
    return samples_all



class DeepSolidEvaluator():
    def __init__(
            self,
            log_dir: str,
            sampling_cfg_str: str,
            libcu_lib_path: str = None,
            dist: bool = False,
            coord_address: str = None,
            num_processes: int = 1,
            job_id: int = None,
            process_id: int = 0,
            timeout: int = None,
            ckpt_restore_filename: str = None,
            x64: bool = False,
        ):
        '''
            config and whether to run in distributed mode are predetermined by the checkpoints saved in log_dir.
        '''
