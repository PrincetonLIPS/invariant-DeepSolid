#!/bin/bash

# helper functions
wait_for_coordinator() {
    counter=$1
    echo "--------------------------------------------------"
    echo "waiting for coordinator, countdown $counter second(s)"
    until [[ $counter < 0 ]]; do
        if [ -f "$2/_coord.ip" ]; then 
            echo "coordinator found"
            break
        fi
        sleep 1
        counter=$(( counter - 1 ))
        echo "waiting for coordinator, countdown $counter second(s)"
    done
    echo "--------------------------------------------------"
    if [ -f "$2/_coord.ip" ]; then
        return 0
    else
        return 1
    fi
}

# print current node status
IPs=$(hostname -I)
IP="${IPs%% *}"
echo "directory for logging: $LOGDIR"
echo "environment activation command: $ENV_CMD"
echo "python command: $PY_CMD"
echo "current IP: $IP"
echo "current node name: $SLURM_JOB_NODELIST"
echo "job id: $SLURM_ARRAY_JOB_ID"
echo "process id: $SLURM_ARRAY_TASK_ID" 
echo "total number of processes: $NUM_JOBS"
echo "timeout: $TIMEOUT"

echo ""
nvidia-smi
echo ""

if [[ $SLURM_ARRAY_TASK_ID = 0 ]]; then 
    # assign coordinator node
    echo "coordinator: true"
    echo "coordination port: $PORT"
    echo "$IP" > "$LOGDIR/_coord.ip"
    export COORD_IP=$IP
else 
    # retrieve coordinator node
    echo "coordinator: false"
    wait_for_coordinator $TIMEOUT $LOGDIR
    if [ -f "$LOGDIR/_coord.ip" ]; then
        export COORD_IP=$(cat "$LOGDIR/_coord.ip")
        echo "coordinator ip: $COORD_IP"
    else 
        echo "coordinator not found. exiting."
        exit 1
    fi
fi

echo "===================================="

# execute script
eval ${ENV_CMD} 
wait
eval ${PY_CMD}
