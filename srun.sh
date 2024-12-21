#!/bin/bash

# Check if two arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <time> <gpu>"
    echo "time: integer between 0 and 60"
    echo "gpu: one of r28t, r39, r49, tr, v100"
    exit 1
fi

# Extract arguments
time=$1
gpu_param=$2

# Validate 'time' (must be an integer between 0 and 60)
if ! [[ $time =~ ^[0-9]+$ ]] || [ $time -lt 0 ] || [ $time -gt 60 ]; then
    echo "Error: 'time' must be an integer between 0 and 60."
    exit 1
fi

# Pad 'time' with a leading zero if less than 10
if [ $time -lt 10 ]; then
    time="0$time"
fi

# Validate 'gpu_param' and convert to appropriate GPU type
case $gpu_param in
    r28t)
        gpu="rtx_2080_ti"
        ;;
    r39)
        gpu="rtx_3090"
        ;;
    r49)
        gpu="rtx_4090"
        ;;
    tr)
        gpu="titan_rtx"  # Assuming this is the value intended for `tr`
        ;;
    v100)
        gpu="v100"
        ;;
    *)
        echo "Error: 'gpu' must be one of r28t, r39, r49, tr, v100."
        exit 1
        ;;
esac

# Construct the srun command
srun_cmd="srun -A ls_hilli -n 4 --mem-per-cpu=4000 --time=00:$time:00 --gpus=$gpu:1 --pty bash"

# Execute the srun command
echo "Executing command: $srun_cmd"
$srun_cmd