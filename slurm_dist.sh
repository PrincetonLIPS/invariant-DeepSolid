#!/bin/bash
set -e

# default variables
run_name=""
file=""
mem="2G"
N="2"
type="gpu"
gres="gpu:1"
port="1234"
log_dir=""
timetout="300"
partition=""
env_cmd=""
py_cmd=""
exclude=""
exclusive=""
A=""
gres="gpu:1"
name="ds"
dependency=""
extra=""

# function to display helper info
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help          Display this help message"
 echo " --env-cmd           Specify command for activating the environment. Need to be a string wrapped in parentheses. Required. Example: \"source ~/miniconda3/bin/activate deepsolid\"." 
 echo " --py-cmd            Specify command for python execution. Need to be a string wrapped in parentheses. Required. Example: \"python train.py\"." 
 echo " --log               Specify the folder for logging. Any existing \"_coord.ip\" file in this folder will be REPLACED. Required" 
 echo " -m, --mem           Specify memory per node. Default to \"2G\""
 echo " -N, --num-nodes     Specify the number of nodes. Default to \"2\". "
 echo " -A                  Specify the allocation for sbatch. Default to \"\"."
 echo " --port              Specify the port number for communication. Default to \"1234\""
 echo " -p, --partition     Specify the partition used in srun"
 echo " --timeout           Specify the timeout for communication in seconds. Default to \"300\""
 echo " --exclude           Specify the nodes to exclude from sbatch. Default to \"\""
 echo " --gres              Specify the resources required per node. Default to \"gpu:1\""
 echo " --name              Specify job name. Default to \"ds\""
 echo " --dependency        Specify dependency. Default to \"\""
 echo " --extra             Specify any extra flags. Default to \"\""
 echo ""
 echo "Variables made available to slurm job file:"
 echo "  - \$LOGDIR                 stores --log argument."
 echo "  - \$PORT                   stores --port argument"
 echo "  - \$NUM_JOBS               stores --num-nodes argument."
 echo "  - \$TIMEOUT                stores --timeout argument."
 echo "  - \$SLURM_ARRAY_TASK_ID    integer to differentiate the tasks."
}

# read options
options=$(getopt -o hmpNA: --long help,exclusive,env-cmd:,py-cmd:,log:,mem:,num-nodes:,port:,partition:,timeout:,exclude:,gres:,name:,dependency:,extra: -- "$@")
eval set -- "$options"

while true; do
  case "$1" in
    -h | --help ) 
        usage; break 
        ;;
    --env-cmd ) 
        env_cmd=$2; shift 2
        ;;
    --py-cmd ) 
        py_cmd=$2; shift 2
        ;;
    --log )
        log_dir=$2; shift 2
        ;;
    -m | --mem ) 
        mem=$2; shift 2 
        ;;
    -N | --num-nodes )
        N=$2; shift 2
        ;;
    -A )
        A="-A $2"; shift 2
        ;;
    --gres )
        gres=$2; shift 2
        ;;
    --name )
        name=$2; shift 2
        ;;
    -p | --partition )
        partition="--partition=$2"; shift 2
        ;;
    --port )
        port=$2; shift 2
        ;;
    --timeout )
        timeout=$2; shift 2
        ;;
    --exclude )
        exclude=$2; shift 2
        ;;
    --dependency )
        dependency="--dependency=$2"; shift 2
        ;;
    --extra )
        extra=$2; shift 2
        ;;
    -- ) 
        shift
        break 
        ;;
    * )
        echo "Invalid option: $1, exiting."
        usage
        exit 1
        break
        ;;
  esac
done

if [[ $log_dir = "" ]]; then 
    echo "logging directory not specified in --log, exiting."
    usage 
    exit 1
fi

# if [[ $env_cmd = "" ]]; then 
#     echo "WARNING: --env-cmd not specified."
# fi

if [[ $py_cmd = "" ]]; then 
    echo "--py-cmd not specified, exiting."
    usage 
    exit 1
fi

# remove existing _coord.ip
if [ -f "$log_dir/_coord.ip" ]; then 
    rm "$log_dir/_coord.ip" 
fi
wait


# sbatch
echo "sbatch $N nodes at port $port with $mem memory per node..."
mkdir -p $log_dir
sbatch --array="0-$((N-1))" --mem=$mem $partition --gres=$gres -o $log_dir/_%A_%a_slurm.out \
       --exclude="$exclude" $A --job-name=$name $dependency $extra \
        --export=NUM_JOBS=$N,PORT=$port,LOGDIR=$log_dir,MEM=$mem,TIMEOUT=$timeout,ENV_CMD="$env_cmd",PY_CMD="$py_cmd" \
        ./utils_slurm/slurm_run.sh