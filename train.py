"""
    Train DeepSolid network.

    Run `python train --help` to see the list of allowed options.
"""
import os
import argparse
from utils.loader import mpatch_load_cfg

def train(
        log_dir: str,
        libcu_lib_path: str = None,
        resume: bool = False,
        dist: bool = False,
        base_cfg_str: str = 'base_config.py',
        cfg_str: str = None,
        optim_cfg_str: str = None,
        coord_address: str = None,
        num_processes: int = 1,
        job_id: int = None,
        process_id: int = 0,
        timeout: int = None,
        x64: bool = False,
    ):
    '''
        Function to execute deepsolid training
    '''
    cfg = mpatch_load_cfg(
        log_dir=log_dir,
        mode='train',
        libcu_lib_path=libcu_lib_path,
        resume=resume,
        dist=dist,
        base_cfg_str=base_cfg_str,
        cfg_str=cfg_str,
        optim_cfg_str=optim_cfg_str,
        coord_address=coord_address,
        num_processes=num_processes,
        job_id=job_id,
        process_id=process_id,
        timeout=timeout,
        x64=x64
    )
    # important to import process only after mpatch
    from utils.process import process
    process(cfg, process_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--libcu_lib_path', type=str, 
                        help='path to lib folder for cuda library files. This is typically the lib folder under the deepsolid env folder.',
                        default=None)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, 
                            help='whether to resume training in an existing folder. "__metadata.pk" will be used for resuming training. --dist, --base_cfg, --cfg and __optim_cfg will be automatically specified.', 
                            default=False)
    parser.add_argument('--dist', action=argparse.BooleanOptionalAction, 
                            help='whether to run in distributed mode', default=False)
    parser.add_argument('--base_cfg', type=str, help='base config file to be copied from ./utils folder', default='base_config.py')
    parser.add_argument('--cfg', type=str, help='config file to be copied from ./config folder', default=None)
    parser.add_argument('--optim_cfg', type=str, help='optimisation config file to be copied from ./optim_config folder', default=None)
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
    
    train(
        log_dir=os.environ["LOGDIR"] + '/',
        libcu_lib_path=args.libcu_lib_path,
        resume=args.resume,
        dist=args.dist,
        base_cfg_str=args.base_cfg,
        cfg_str=args.cfg,
        optim_cfg_str=args.optim_cfg,
        coord_address=coord_address,
        num_processes=num_processes,
        process_id=process_id,
        job_id=job_id,
        timeout=timeout,
        x64=args.x64
    )