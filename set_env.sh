#!/bin/bash

# enter smplx directory
cd "$gs" || { echo "Error: Directory $gs does not exist."; exit 1; }


# Load CUDA module
module load cuda/12.2 || { echo "Error: Failed to load CUDA module 12.2."; exit 1; }

# activate conda env
source /local/home/lingxi/data_lx/anaconda3/etc/profile.d/conda.sh
conda activate 3dgs-avatar || { echo "Error: Failed to activate conda environment '3dgs-avatar'."; exit 1; }

echo "Environment setup successfully completed."