#!/bin/sh


#SBATCH --time=2-00:00:00
#SBATCH --partition=cpuonly
#SBATCH --job-name=w2v-mlp-arxiv2023
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

data="arxiv_2023"
max_iter=1000
embedders=("w2v" "tfidf" "original")
decoders=("dot" "concat" "euclidean")
devices=("cuda:2" "cuda:0" "cuda:1")

for embedder in "${embedders[@]}"; do
    for i in ${!decoders[@]}; do
        decoder=${decoders[$i]}
        device=${devices[$i]}
        echo "python lp_node_embed.py --data $data --embedder $embedder --device $device --decoder $decoder --epoch $max_iter"
        python lp_node_embed.py --data $data --embedder $embedder --device $device --decoder $decoder --epoch $max_iter &
    done
    wait 
done

# for i in ${!decoders[@]}; do
#     decoder=${decoders[$i]}
#     device=${devices[$i]}
#     echo "python lp_edge_embed.py --data $data --embedder $embedder --device $device "
#     python lp_edge_embed.py --data $data --embedder $embedder --device $device --epochs $max_iter
# done
