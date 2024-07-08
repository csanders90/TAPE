#!/bin/sh

#SBATCH --time=3-00:00:00
#SBATCH --partition=cpuonly 
#SBATCH --job-name=tfidf-mlp-pubmed


#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/batch

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu
source /hkfs/home/project/hk-project-test-p0022257/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate TAG-LP
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/model_finetuning


for iter in 1500; do
    echo "python mlp.py --data pubmed --decoder MLP --max_iter $iter"
    python mlp.py --data pubmed --decoder MLP --max_iter $iter
done
