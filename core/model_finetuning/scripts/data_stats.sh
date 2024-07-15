#!/bin/sh
#SBATCH --time=3-00:00:00
#SBATCH --partition=cpuonly
#SBATCH --job-name=tfidf-mlp-ogbn_arxiv



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

# test 

# for iter in 2 10 20 30 40; do
#     echo "python mlp.py --data ogbn-arxiv --decoder MLP --max_iter $iter"
#     python mlp.py --data ogbn-arxiv --decoder MLP --max_iter $iter
# done

# python adj.py --data pwc_large --scale 1000 > pwc_large.output
# python adj.py --data citationv8 --scale 1000 > citationv8.output
# python adj.py --data ogbn-arxiv --scale 1000 > ogbn-arxiv.output
# python adj.py --data pwc_medium --scale 1000 > pwc_medium.output
# python adj.py --data pubmed --scale 1000 > pubmed.output
# python adj.py --data arxiv_2023  > arxiv_2023.output
# python adj.py --data cora > cora.output
# python adj.py --data pwc_small > pwc_small.output