#!/bin/bash

# Change to the directory containing your bash scripts
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/embedding_mlp/scripts

# Loop through each .sh file in the directory and submit it using sbatch
for script in *.sh; do
    sbatch "$script"
done

squeue --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" -u cc7738