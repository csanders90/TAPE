#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb
#SBATCH --mem-per-cpu=1600mb
#SBATCH --gres=gpu:full:4, 
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate nui
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns

# Define the list of items for the loop
items=(ogbn-arxiv ogbn-product pubmed ogbn-products)

# Run the loop in parallel
for item in ogbn-arxiv ogbn-products pubmed ogbn-products; do
    echo "Processing $item"
    python custom_tune.py --data "$item" > "$item-VAE-custom_tune_output.txt" 
done 
