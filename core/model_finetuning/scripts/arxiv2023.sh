#!/bin/sh


#SBATCH --time=3-00:00:00
#SBATCH --partition=cpuonly 
#SBATCH --job-name=w2v-mlp-arxiv2023
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu

source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate TAG-LP
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12



cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/model_finetuning

data="arxiv_2023"
max_iter=10000
embedder="w2v"
# for iter in 1000 2000 10000; do
#     echo "python mlp.py --data arxiv_2023 --decoder MLP --max_iter $iter"
#     python mlp.py --data arxiv_2023 --decoder MLP --max_iter $iter
# done

echo "python mlp.py --data $data --max_iter $max_iter --embedder $embedder"
python mlp.py --data $data --max_iter $max_iter --embedder $embedder
