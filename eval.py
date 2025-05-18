"""
    Evaluate a DeepSolid network on a given list of electron coordinates

    Run `python eval --help` to see the list of allowed options.

    Example for generating an input file. Note that ts needs to be compatible with SpaceGroupVisualizer.plot_evals in utils.vizualize
    >>>>>>
        from utils.loader import load_module
        import numpy as np

        base_cfg = load_module('base_cfg', 'utils/base_config.py').default()
        cfg = load_module('cfg', 'config/graphene_1.py').get_config(base_cfg)
        L_Bohr = cfg.system.pyscf_cell.L_Bohr
        elecs, _ = load_module('symmscan', f'symmscan_config/graphene-1.py').symm_cfg(L_Bohr)

        tx = ty = np.linspace( - 2 * L_Bohr, 2 * L_Bohr, 500)
        X, Y = np.meshgrid(tx, ty)
        translates = np.array([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())]).T

        ts = translates[:,:2]
        xs = np.array([ (elecs + t).flatten() for t in translates]) # simultaneous translations
        np.savez(f'{log_dir}symmscan_-2_2_500.npz', xs=xs, ts=ts)
    >>>>>>

"""

import argparse, os, sys
import numpy as np
from utils.loader import mpatch_load_cfg, load_module
import datetime

from absl import logging
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

def retrieve_evals(
        log_dir: str,
        mode: str,
        input_file: str,
        ckpt_restore_filename: str,
        num_processes: int = 1,
    ):
        assert mode in ['slogdet', 'energy']
        x_list = []
        t_list = []
        val_list = []
        for process_id in range(num_processes):
            fname = f"{log_dir}{ckpt_restore_filename.split('.npz')[0]}_{input_file.split('.npz')[0]}_{mode}_{process_id}.npz"
            output = np.load(fname)
            x_list.append( output['xs'].reshape((-1, output['xs'].shape[-1])) ) 
            t_list.append( output['ts'].reshape((-1, output['ts'].shape[-1])) )
            val_list.append( output['vals'].reshape((-1)) )
        xs = np.concatenate(x_list, axis=0)
        ts = np.concatenate(t_list, axis=0)
        vals = np.concatenate(val_list, axis=0)
        return xs, ts, vals

def eval(
        log_dir: str,
        mode: str,
        input_file: str,
        ckpt_restore_filename: str,
        eval_cfg_str: str = None,
        batch_size: int = None,
        libcu_lib_path: str = None,
        num_processes: int = 1,
        process_id: int = 0,
        save_freq: int = 10,
        x64: bool = False,
    ):
    '''
        Function to evaluate a given deepsolid network
    '''
    assert mode in ['slogdet', 'energy']
    # load network
    params, cfg, sharded_key, _, _, _, batch_slater_slogdet, total_energy = mpatch_load_cfg(
            log_dir=log_dir,
            mode='eval',
            resume=True,
            ckpt_restore_filename=ckpt_restore_filename,
            x64=x64,
            libcu_lib_path=libcu_lib_path,
            eval_cfg_str=eval_cfg_str,
    )
    # load input
    input = np.load(f'{log_dir}{input_file}')
    xs = input['xs'] #actual input
    ts = input['ts'] #variable for plotting
    
    assert len(xs.shape) == 2 and xs.shape[1] == 3 * np.sum(cfg.system.pyscf_cell.nelec)
    assert xs.shape[0] % num_processes == 0
    assert len(ts.shape) == 2 and ts.shape[1] in [1,2] and ts.shape[0] == xs.shape[0]
    
    # evaluation
    import jax 
    import jax.numpy as jnp
    from DeepSolid import constants
    from utils.addkeys import split_data_keys, pad_data_with_key

    total_energy = constants.pmap(jax.jit(total_energy))
    batch_slater_slogdet = constants.pmap(jax.jit(batch_slater_slogdet))

    # for parallel processing, keep only the inputs allocated for this device
    xs = jnp.array(xs).reshape((num_processes,-1,xs.shape[-1]))
    ts = jnp.array(ts).reshape((num_processes,-1,ts.shape[-1]))

    # evaluate in batches
    if batch_size is None:
        batch_size == xs.shape[1]
    num_batches = (xs.shape[1] - 1) // batch_size + 1

    x_list = []
    t_list = []
    val_list = []
    fname = f"{log_dir}{ckpt_restore_filename.split('.npz')[0]}_{input_file.split('.npz')[0]}_{mode}_{process_id}.npz"
    logging.info(f"====================\n Evaluating wavefunction... Saving to {fname}...\n====================")
    for b in range(num_batches):
        logging.info( f'{datetime.datetime.now():%Y %b %d %H:%M:%S} batch no. {b+1} / {num_batches} of {batch_size} samples')
        x_batch = xs[:,b*batch_size:min((b+1)*batch_size,xs.shape[1]),:]
        x_list.append( x_batch )
        t_batch = ts[:,b*batch_size:min((b+1)*batch_size,xs.shape[1]),:]
        t_list.append( t_batch )

        if cfg.symmetria.gpave.on:
            sharded_key, dkeys = constants.p_split(sharded_key)
            data_with_keys = pad_data_with_key(x_batch, dkeys)
            if mode == 'energy':
                val_list.append( total_energy(params, data_with_keys)[1]['local_energy'] )
            else:
                data, keys = jax.vmap(lambda a: split_data_keys(a))(data_with_keys)
                val_list.append( batch_slater_slogdet( params, data, keys ) )
        else:            
            if mode == 'energy':
                val_list.append( total_energy(params, x_batch)[1]['local_energy'] )
            else:
                val_list.append( batch_slater_slogdet( params, x_batch ) )
        
        if b == num_batches-1 or b % save_freq == 0:
            np.savez(fname, xs=np.array(x_list), ts=np.array(t_list), vals=np.array(val_list))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='either slogdet or energy', required=True)
    parser.add_argument('--input_file', type=str, help='npz file containing an n x 3N array that represents the list of all configurations (each consisting of N electrons in 3d) to evaluate. Note that n must be divisible by number of processes', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size for sequential processing. By default, process all inputs in one batch.', default=None)
    parser.add_argument('--libcu_lib_path', type=str, 
                        help='path to lib folder for cuda library files. This is typically the lib folder under the deepsolid env folder.',
                        default=None)
    parser.add_argument('--ckpt_restore_filename', type=str, help='checkpoint file to be restored', default=None)
    parser.add_argument('--eval_cfg_str', type=str, help='file with additional cfg for evals', default=None)
    parser.add_argument('--x64', action=argparse.BooleanOptionalAction, 
                            help='whether to enable x64.', 
                            default=False)
    parser.add_argument('--save_freq', type=int, help='save frequency (in number of batches)', default=10)
    args = parser.parse_args()

    # read environmental variables for distributed setup
    if 'COORD_IP' in os.environ and 'PORT' in os.environ:
        coord_address = str(os.environ['COORD_IP']).strip()+":"+str(os.environ['PORT']).strip()
    else:
        coord_address = None
    
    num_processes_str = os.environ.get('NUM_JOBS')
    num_processes = int(num_processes_str) if num_processes_str else 1
    
    process_id_str = os.environ.get('SLURM_ARRAY_TASK_ID')
    process_id = int(process_id_str) if process_id_str else 0

    eval(
        log_dir=os.environ["LOGDIR"] + '/',
        libcu_lib_path=args.libcu_lib_path,
        input_file=args.input_file,
        mode=args.mode,
        batch_size=args.batch_size,
        eval_cfg_str=args.eval_cfg_str,
        num_processes=num_processes,
        process_id=process_id,
        ckpt_restore_filename=args.ckpt_restore_filename,
        save_freq=args.save_freq,
        x64=args.x64
    )