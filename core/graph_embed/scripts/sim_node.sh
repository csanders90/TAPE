#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --partition=cpuonly
#SBATCH --job-name=tag_struc2vec
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmar_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/scripts

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate TAG-LP
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/core/textfeat
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


ls -ltr

for VARIABLE in 1 2 3 4 5 .. 10
do
    python mlp_dot_product.py --cfg core/yamls/cora/struc2vec.yaml 
done

for VARIABLE in 1 2 3 4 5 .. 10
do
    python  mlp_dot_product.py --cfg core/yamls/arxiv_2023/struc2vec.yaml 
done

for VARIABLE in 1 2 3 4 5 .. 10
do
    python  mlp_dot_product.py --cfg core/yamls/pubmed/struc2vec.yaml 
done