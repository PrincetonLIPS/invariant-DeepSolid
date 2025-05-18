"""
    Sample DeepSolid network.

    Run `python sampling --help` to see the list of allowed options.
"""
import os
import argparse
from utils.sampler import DeepSolidSampler

def sampling(
        log_dir: str,
        sampling_cfg_str: str,
        required_samples: int,
        libcu_lib_path: str = None,
        dist: bool = False,
        coord_address: str = None,
        num_processes: int = 1,
        job_id: int = None,
        process_id: int = 0,
        timeout: int = None,
        ckpt_restore_filename: str = None,
        save_freq: int = 10,
        x64: bool = False,
    ):
    '''
        Function to execute deepsolid sampling
    '''
    sampler = DeepSolidSampler(
                log_dir=log_dir,
                sampling_cfg_str=sampling_cfg_str,
                libcu_lib_path=libcu_lib_path,
                dist=dist,
                coord_address=coord_address,
                num_processes=num_processes,
                job_id=job_id,
                process_id=process_id,
                timeout=timeout,
                ckpt_restore_filename=ckpt_restore_filename,
                x64=x64
    )
    sampler.load_samples()
    sampler.draw_samples(required_samples=required_samples, save_freq=save_freq)
    return sampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--libcu_lib_path', type=str, 
                        help='path to lib folder for cuda library files. This is typically the lib folder under the deepsolid env folder.',
                        default=None)
    parser.add_argument('--dist', action=argparse.BooleanOptionalAction, 
                            help='whether to run in distributed mode', default=False)
    parser.add_argument('--sampling_cfg', type=str, help='sampling config file to be copied from ./sampling_cfg folder', required=True)
    parser.add_argument('--required_samples', type=int, help='number of samples required', required=True)
    parser.add_argument('--save_freq', type=int, help='save frequency (in number of batches)', default=10)
    parser.add_argument('--ckpt_restore_filename', type=str, help='checkpoint file to be restored', default=None)
    parser.add_argument('--x64', action=argparse.BooleanOptionalAction, 
                            help='whether to enable x64.', 
                            default=False)
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

    job_id_str = os.environ.get('SLURM_ARRAY_JOB_ID')
    job_id = int(job_id_str) if job_id_str else None

    timeout_str = os.environ.get('TIMEOUT')
    timeout = int(timeout_str) if timeout_str else None
    
    sampling(
        log_dir=os.environ["LOGDIR"] + '/',
        libcu_lib_path=args.libcu_lib_path,
        sampling_cfg_str=args.sampling_cfg,
        required_samples=args.required_samples,
        dist=args.dist,
        coord_address=coord_address,
        num_processes=num_processes,
        process_id=process_id,
        job_id=job_id,
        timeout=timeout,
        ckpt_restore_filename=args.ckpt_restore_filename,
        save_freq=args.save_freq,
        x64=args.x64
    )